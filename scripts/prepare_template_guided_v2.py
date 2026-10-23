"""CPU-only V2 identity/formula checks and exhaustive dual-fingerprint audit."""

import gzip
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rdkit import Chem, DataStructs
from scipy.stats import spearmanr

from structure_pretrain.data.prepare_v03 import atom_counts
from structure_pretrain.formal_v2.common import (
    CONFIG,
    array_digest,
    configuration,
    dedicated_root,
    digest,
    emit,
    read,
    verify_data,
    write,
)
from structure_pretrain.formal_v2.retrieval import exact_retrieve
from structure_pretrain.polar_only_v04.chemistry import chainless_molecule, derive_polar_formula
from structure_pretrain.polar_only_v04.features import packed_fingerprint
from structure_pretrain.reported_headgroups.chemistry import builder_trial
from structure_pretrain.reported_headgroups.formal_template_data import FEATURES, V03
from structure_pretrain.reported_headgroups.template_selection import (
    MORGAN,
    phosphate_comparison_key,
    read_csv,
)
from structure_pretrain.utils.formula import FormulaStandardizer


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
        **{f"fraction_at_least_{x}": float(np.mean(values >= x)) for x in (0.2, 0.4, 0.6, 0.8)},
    }


def main():
    root = dedicated_root()
    c = configuration()
    torch.set_num_threads(4)
    data = root / "data"
    audit = root / "audit"
    data.mkdir(exist_ok=True)
    audit.mkdir(exist_ok=True)
    if (audit / "A_passed.json").exists() and read(audit / "A_passed.json").get(
        "query_binding_version"
    ) == 2:
        verify_data(root)
        emit(stage="A_already_complete", report=read(audit / "A_passed.json"))
        return
    began = time.monotonic()
    old = Path(c["shared_exploratory_root"]) / "data"
    manifest = read(old / "manifest.json")
    assert manifest["status"] == "passed" and manifest["split_counts"] == c["split_counts"]
    immutable = {p: digest(p) for p in manifest["frozen_input_sha256"]}
    assert immutable == manifest["frozen_input_sha256"]
    assert all(digest(old / name) == sha for name, sha in manifest["data_assets_sha256"].items())
    # Reuse physical input files; never write through a shared hard link.
    for name in (
        "tokens",
        "token_lengths",
        "splits",
        "target_rows",
        "full_formulas",
        "chain_labels",
        "polar_formulas",
        "occupancy",
        "bank_packed",
        "bank_formulas",
        "diagnostic_512_original_rows",
        "diagnostic_512_groups",
    ):
        target = data / (name + ".npy")
        source = old / (name + ".npy")
        if not target.exists():
            os.link(source.resolve(), target)
        assert digest(target) == digest(source)
    pqheads = pq.read_table(
        V03 / "polar_groups.parquet",
        columns=["group", "canonical_polar_smiles", "split", "chemistry_family"],
    ).to_pydict()
    labels = np.asarray(pqheads["canonical_polar_smiles"], dtype=object)
    assert np.array_equal(pqheads["group"], np.arange(len(labels)))
    by_smiles = {s: i for i, s in enumerate(labels)}
    library = pq.read_table(
        "data/structure_library_v0.3/polar_library.parquet",
        columns=["polar_id", "polar_fragment_smiles"],
    ).to_pydict()
    with np.load(
        "outputs/headgroup_template_proposal_20261022/selection_cache_private.npz"
    ) as cache:
        packed, oldkeys, masks = cache["packed"], cache["keys"], cache["masks"]
    order = np.asarray([by_smiles[s] for s in library["polar_fragment_smiles"]], dtype=np.int64)
    polar_fp = np.empty_like(packed)
    polar_fp[order] = packed
    keys = np.empty(len(labels), dtype=oldkeys.dtype)
    keys[order] = oldkeys
    group_masks = np.empty_like(masks)
    group_masks[order] = masks
    del packed, oldkeys, masks
    frozen_split = np.asarray(
        [{"train": 0, "validation": 1, "test": 2}[x] for x in pqheads["split"]], dtype=np.uint8
    )
    splits = np.load(data / "splits.npy")
    scope = splits != 255
    eligible = np.flatnonzero(scope)
    sets = {i: set(keys[splits == i]) for i in (0, 1, 2)}
    intersections = {f"{i}_{j}": len(sets[i] & sets[j]) for i, j in ((0, 1), (0, 2), (1, 2))}
    assert not any(intersections.values())
    assert all(len(sets[i]) == int((splits == i).sum()) for i in (0, 1, 2)), (
        "Within-split normalized duplicates"
    )
    all_groups = np.load(V03 / "polar_groups.npy", mmap_mode="r")
    row_splits = np.load(V03 / "row_splits.npy", mmap_mode="r")
    assert np.array_equal(frozen_split[all_groups], row_splits), (
        "Different chains leaked across splits"
    )
    full = np.load(data / "full_formulas.npy")
    chains = np.load(data / "chain_labels.npy")
    polar = np.load(data / "polar_formulas.npy")
    np.testing.assert_array_equal(
        derive_polar_formula(
            full,
            chains,
            np.load(FEATURES / "chain_counts.npy"),
            np.load(FEATURES / "scaffold_counts.npy"),
        ),
        polar,
    )
    selected = read(old / "nested_templates_20000_private.json")[:5000]
    bank_keys = np.asarray([r["comparison_key"] for r in selected])
    bank_ids = np.asarray([r["template_id"] for r in selected])
    assert len(set(bank_keys)) == 5000 and not (set(bank_keys) & (sets[1] | sets[2]))
    approved = {
        r["headgroup_id"]: r
        for r in read_csv(
            "outputs/headgroup_template_proposal_20261022/approved_reference_catalog.csv"
        )
    }
    seed_records = []
    bank_polar = np.empty((5000, 256), dtype=np.uint8)
    expected_bank_formula = np.load(data / "bank_formulas.npy")
    expected_bank_fingerprints = np.load(data / "bank_packed.npy")
    for i, r in enumerate(selected):
        mol = Chem.MolFromSmiles(r["polar_smiles"])
        assert mol is not None
        np.testing.assert_array_equal(atom_counts(mol), expected_bank_formula[i])
        assert builder_trial(r["polar_smiles"])["independent_element_H_balance"]
        for count in (1, 2):
            np.testing.assert_array_equal(
                packed_fingerprint(chainless_molecule(r["polar_smiles"], count)),
                expected_bank_fingerprints[2 - count, i],
            )
        bank_polar[i] = np.frombuffer(
            DataStructs.BitVectToBinaryText(MORGAN.GetFingerprint(mol)), dtype=np.uint8
        )
        if r.get("approved_seed_id"):
            source = approved[r["approved_seed_id"]]
            assert source["builder_compatible"] == "True"
            assert phosphate_comparison_key(source["polar_smiles"]) == r["comparison_key"]
            seed_records.append(
                {
                    "template_id": r["template_id"],
                    "selected_polar_smiles": r["polar_smiles"],
                    "approved_record": source,
                }
            )
        if (i + 1) % 1000 == 0:
            emit(stage="A1_builder_H_audit", templates=i + 1)
    assert len(seed_records) == 118
    with gzip.open(audit / "seed_provenance_private.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in seed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(data / "polar_fingerprints.npy", polar_fp)
    np.save(data / "functional_masks.npy", group_masks)
    np.save(data / "bank_polar_packed.npy", bank_polar)
    # Stable reusable core identity; detailed graphs stay only on G.
    write(data / "template_identity_private.json", selected)
    np.save(data / "comparison_keys.npy", keys)
    write(
        audit / "A1_integrity.json",
        {
            "status": "passed",
            "frozen_cores": len(labels),
            "eligible": len(eligible),
            "split_counts": c["split_counts"],
            "cross_split_normalized_intersections": intersections,
            "all_full_lipid_rows_match_core_split": int(len(all_groups)),
            "formula_identities_checked": len(full),
            "bank_Builder_H_checks": 5000,
            "chainless_bank_fingerprint_checks": 10000,
            "traceable_human_seeds": 118,
            "source_input_sha256": immutable,
            "shared_data_directory": str(old),
            "historical_data_modified": False,
        },
    )
    emit(stage="A1_passed")

    def retrieval_run(name, queries, bank):
        indices_path = audit / (name + "_indices.npy")
        scores_path = audit / (name + "_scores.npy")
        report_path = audit / (name + "_audit.json")
        binding = {
            "query_array_sha256": array_digest(queries),
            "bank_array_sha256": array_digest(bank),
            "query_keys_sha256": array_digest(keys[eligible]),
            "bank_keys_sha256": array_digest(bank_keys),
            "bank_ids_sha256": array_digest(bank_ids),
        }
        if report_path.exists():
            proof = read(report_path)
            assert (
                digest(indices_path) == proof["indices_sha256"]
                and digest(scores_path) == proof["scores_sha256"]
            )
            if proof.get("query_binding_version") != 2:
                # One-time provenance upgrade: independently recheck cached IDs and
                # scores against full-bank RDKit using a different random sample.
                from structure_pretrain.formal_v2.retrieval import bitvectors, oracle

                vectors = bitvectors(bank)
                tie_order = np.empty(len(bank), dtype=np.int64)
                tie_order[np.argsort(bank_ids, kind="stable")] = np.arange(len(bank))
                idx, scores = np.load(indices_path), np.load(scores_path)
                sample = np.random.default_rng(20261024).choice(len(queries), 512, replace=False)
                for row in sample:
                    ids, values = oracle(
                        queries[row], vectors, bank_keys, keys[eligible[row]], tie_order, 3
                    )
                    np.testing.assert_array_equal(ids, idx[row])
                    np.testing.assert_allclose(values, scores[row], atol=1e-6, rtol=0)
                proof.update(
                    binding, query_binding_version=2, provenance_upgrade_oracle_matches=512
                )
                proof.pop("query_sha256", None)
                proof.pop("bank_sha256", None)
                write(report_path, proof)
            assert all(proof[name] == value for name, value in binding.items())
            return np.load(indices_path), np.load(scores_path), proof
        idx, sim, proof = exact_retrieve(
            queries,
            bank,
            keys[eligible],
            bank_keys,
            bank_ids,
            batch_size=c["audit"]["query_batch"],
            audit_count=c["audit"]["exact_oracle_sample"],
            event=emit,
        )
        np.save(indices_path, idx)
        np.save(scores_path, sim)
        proof.update(
            binding,
            query_binding_version=2,
            indices_sha256=digest(indices_path),
            scores_sha256=digest(scores_path),
        )
        write(report_path, proof)
        return idx, sim, proof

    pidx, psim, paudit = retrieval_run("polar", polar_fp[eligible], bank_polar)
    frozen_fp = np.load(FEATURES / "chainless_fingerprints.npy", mmap_mode="r")
    lookup = np.full((len(labels), 2), -1, dtype=np.int32)
    lookup[
        np.load(FEATURES / "variant_heads.npy"), 2 - np.load(FEATURES / "variant_occupancy.npy")
    ] = np.arange(len(frozen_fp))
    assert np.all(lookup >= 0)
    bank_fp = np.load(data / "bank_packed.npy")
    newrefs = np.full((len(labels), 2, 3), -1, dtype=np.int32)
    newsims = np.zeros((len(labels), 2, 3), dtype=np.float32)
    comparisons = {}
    audits = {"polar": paudit}
    oldrefs = np.load(old / "references.npy", mmap_mode="r")
    rows = {
        "group": eligible,
        "split": splits[eligible],
        "occupancy": np.load(data / "occupancy.npy")[eligible],
        "polar_top1_index": pidx[:, 0],
        "polar_top1_score": psim[:, 0],
    }
    for occupancy in (0, 1):
        idx, sim, proof = retrieval_run(
            "chainless_" + str(occupancy),
            np.asarray(frozen_fp[lookup[eligible, occupancy]]),
            bank_fp[occupancy],
        )
        audits["chainless_" + str(occupancy)] = proof
        newrefs[eligible, occupancy] = idx
        newsims[eligible, occupancy] = sim
        same = idx[:, 0] == pidx[:, 0]
        overlap = np.asarray(
            [len(set(a) & set(b)) / 3 for a, b in zip(pidx, idx)], dtype=np.float32
        )
        comparisons[str(occupancy)] = {
            "occupancy": "double" if occupancy == 0 else "single",
            "queries": "same eligible Polar cores with this fixed glycerol occupancy",
            "N": len(eligible),
            "top1_identity_agreement": float(same.mean()),
            "mean_top3_intersection_over3": float(overlap.mean()),
            "top1_score_Spearman": float(spearmanr(psim[:, 0], sim[:, 0]).statistic),
            "polar_top3_mean_scores": psim.mean(0).tolist(),
            "chainless_top3_mean_scores": sim.mean(0).tolist(),
            "top1_changed_vs_exploratory_HNSW": int(
                np.count_nonzero(idx[:, 0] != oldrefs[eligible, occupancy, 0])
            ),
            "top3_overlap_distribution": dict(Counter(int(round(x * 3)) for x in overlap)),
            "by_split": {
                name: {
                    "polar": distribution(psim[splits[eligible] == j, 0]),
                    "chainless": distribution(sim[splits[eligible] == j, 0]),
                }
                for j, name in enumerate(("train", "validation", "test"))
            },
        }
        rows["chainless_" + str(occupancy) + "_top1_index"] = idx[:, 0]
        rows["chainless_" + str(occupancy) + "_top1_score"] = sim[:, 0]
        rows["chainless_" + str(occupancy) + "_overlap"] = overlap
    newrefs[newsims < c["minimum_similarity"]] = -1
    valid = newrefs[eligible] >= 0
    assert valid.any(-1).all(), "Any zero-reference target is a hard audit failure"
    for o in (0, 1):
        assert not np.any(
            (newrefs[eligible, o] >= 0)
            & (bank_keys[newrefs[eligible, o].clip(min=0)] == keys[eligible, None])
        )
    np.save(data / "references.npy", newrefs)
    np.save(data / "reference_similarities.npy", newsims)

    def pairs():
        train = np.flatnonzero(splits == 0)
        occupancy = np.load(data / "occupancy.npy")
        for start in range(0, len(train), 4096):
            g = train[start : start + 4096]
            refs = newrefs[g, occupancy[g]]
            mask = refs >= 0
            f = expected_bank_formula[refs.clip(min=0)]
            yield {
                "template_formulas": torch.tensor(f),
                "delta_formulas": torch.tensor(polar[g, None] - f),
                "template_mask": torch.tensor(mask),
            }

    stats = FormulaStandardizer.fit(pairs())
    stats.update(training_pairs_only=True, K=3, bank_size=5000)
    write(data / "formula_statistics.json", stats)
    pq.write_table(
        pa.Table.from_pydict(rows),
        audit / "dual_fingerprint_per_core_private.parquet",
        compression="zstd",
    )
    write(audit / "A2_dual_fingerprint.json", comparisons)
    write(
        audit / "A3_retrieval.json",
        {
            "status": "passed",
            "threshold": c["minimum_similarity"],
            "methods": audits,
            "mask_count_by_occupancy": {
                str(o): dict(Counter(int(x) for x in valid[:, o].sum(-1))) for o in (0, 1)
            },
            "no_silent_target_exclusions": True,
            "low_similarity_top1_lt04_by_occupancy": {
                str(o): int((newsims[eligible, o, 0] < 0.4).sum()) for o in (0, 1)
            },
        },
    )
    assets = {
        p.name: digest(p)
        for p in sorted(data.glob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    write(
        data / "manifest.json",
        {
            "status": "passed",
            "query_binding_version": 2,
            "schema": c["schema"],
            "original_exploratory_manifest_sha256": digest(old / "manifest.json"),
            "configuration_sha256": digest(CONFIG),
            "data_assets_sha256": assets,
            "frozen_input_sha256": immutable,
            "split_counts": c["split_counts"],
            "references": "exhaustive exact Tanimoto, same frozen5000, own/equivalent excluded",
            "statistics": (
                "refit from TRAIN pairs only after exact retrieval; original "
                "arrays/labels untouched"
            ),
        },
    )
    write(
        audit / "A_passed.json",
        {
            "status": "passed",
            "elapsed_seconds": time.monotonic() - began,
            "query_binding_version": 2,
            "data_manifest_sha256": digest(data / "manifest.json"),
            "cluster_diagnostic": "pending independent B diagnostic; required before D",
            "new_training_launched": False,
        },
    )
    emit(stage="A_completed", seconds=time.monotonic() - began)


if __name__ == "__main__":
    main()
