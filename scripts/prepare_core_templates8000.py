"""CPU-only freeze of whole-core labels, grouped splits and a TRAIN-only bank."""

import hashlib
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from structure_pretrain.core_templates.chemistry import equivalence_key, head_fingerprint
from structure_pretrain.core_templates.common import (
    CONFIG,
    ROOT,
    configuration,
    digest,
    emit,
    tokenizer,
    verify_data,
    write,
)
from structure_pretrain.data.smiles_tokenizer import TOKEN_PATTERN, SmilesTokenizer
from structure_pretrain.formal_v2.retrieval import exact_retrieve
from structure_pretrain.glycerol_core_collection.chemistry import capped_core
from structure_pretrain.glycerol_substructure_collection.chemistry import counts, role_frame
from structure_pretrain.reported_headgroups.template_selection import functional_mask
from structure_pretrain.utils.formula import ELEMENTS, FormulaStandardizer


def grouped_split(keys, seed=42, fractions=(0.9, 0.05, 0.05)):
    """Equivalent capped graphs cannot appear on both sides, even with other ports."""
    unique, inverse = np.unique(keys, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    ntrain = int(math.floor(len(unique) * fractions[0]))
    nval = int(math.floor(len(unique) * fractions[1]))
    split = np.full(len(unique), 2, dtype=np.uint8)
    split[order[:ntrain]] = 0
    split[order[ntrain : ntrain + nval]] = 1
    return split[inverse]


def balanced_quotas(sizes, wanted):
    """Bounded integer sqrt-population quotas with deterministic reallocation."""
    sizes = np.asarray(sizes, dtype=np.int64)
    if wanted > sizes.sum():
        raise ValueError("Bank larger than distinct TRAIN structures")
    q = np.zeros_like(sizes)
    while q.sum() < wanted:
        available = np.flatnonzero(q < sizes)
        scores = np.sqrt(sizes[available]) / (q[available] + 1)
        chosen = available[int(np.argmax(scores))]
        q[chosen] += 1
    return q


def summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "quantiles_0_25_50_75_95_99_100": np.quantile(
            values, [0, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
        ).tolist(),
        **{f"fraction_ge_{x}": float(np.mean(values >= x)) for x in (0.2, 0.4, 0.6, 0.8)},
    }


def main():
    c, began = configuration(), time.monotonic()
    root = ROOT.resolve()
    root.mkdir(exist_ok=True)
    folder = root / "data"
    folder.mkdir(exist_ok=True)
    if (folder / "manifest.json").exists():
        verify_data(root)
        emit(stage="prepared_data_already_verified")
        return
    if (root / "gpu_monitor_epoch_private.json").exists():
        raise ValueError("Cannot prepare after the new GPU monitor epoch began")
    RDLogger.DisableLog("rdApp.warning")
    torch.set_num_threads(c["cpu_threads"])
    source = Path(c["library"])
    table = pq.read_table(source)
    assert len(table) == c["expected_library_count"]
    rows = table.to_pylist()
    ids = np.asarray([r["target_id"] for r in rows])
    assert len(set(ids)) == len(rows)
    assert all(r["selection_status"] == "main" and r["assembly_H_formula_verified"] for r in rows)
    all_pieces = [TOKEN_PATTERN.findall(r["target_smiles"]) for r in rows]
    assert all("".join(p) == r["target_smiles"] for p, r in zip(all_pieces, rows))
    vocabulary = sorted(set(x for p in all_pieces for x in p))
    extra = sorted(set(vocabulary) - set(SmilesTokenizer().tokens))
    write(
        folder / "tokenizer.json",
        {
            "extra_tokens": extra,
            "label_scope": c["label_scope"],
            "vocabulary_source": "fixed chemical syntax inventory; no fitted validation statistics",
        },
    )
    tok = tokenizer(root)
    lengths = np.asarray([len(p) + 2 for p in all_pieces], dtype=np.int16)
    assert int(lengths.max()) <= c["max_label_tokens"], "No label truncation/drop allowed"
    tokens = np.full((len(rows), c["max_label_tokens"]), tok.pad_id, dtype=np.int16)
    formulas = np.zeros((len(rows), 10), dtype=np.int16)
    packed = np.zeros((len(rows), 256), dtype=np.uint8)
    masks = np.zeros(len(rows), dtype=np.uint32)
    keys = []
    frame_checks = 0
    for i, r in enumerate(rows):
        mol = Chem.MolFromSmiles(r["target_smiles"])
        assert mol is not None
        match, _, head_c, outside = role_frame(mol)
        assert head_c != match[2]
        assert (outside if mol.GetAtomWithIdx(outside).GetAtomicNum() == 15 else head_c) == 0
        cap = capped_core(mol)
        canonical = Chem.MolToSmiles(cap, canonical=True, isomericSmiles=False)
        assert canonical == r["capped_core_smiles"]
        assert Chem.GetFormalCharge(cap) == 0
        atom_counts = counts(cap)
        assert not (set(atom_counts) - set(ELEMENTS))
        formulas[i] = [atom_counts[e] for e in ELEMENTS]
        tokens[i, : lengths[i]] = tok.encode(r["target_smiles"])
        assert tok.decode(tokens[i, : lengths[i]].tolist()) == r["target_smiles"]
        keys.append(equivalence_key(cap))
        fp = head_fingerprint(mol)
        packed[i] = np.frombuffer(DataStructs.BitVectToBinaryText(fp), dtype=np.uint8)
        masks[i] = functional_mask(cap)
        frame_checks += 1
        if (i + 1) % 5000 == 0:
            emit(stage="whole_core_labels_formulas_fingerprints", targets=i + 1, total=len(rows))
    # Keep fixed-size hashes rather than thousands of variable-length chemical strings
    # in all retrieval comparisons; full grouping keys are in private targets.parquet.
    key_hash = np.asarray([hashlib.sha256(k.encode()).hexdigest() for k in keys])
    assert len(set(key_hash)) == len(set(keys))
    splits = grouped_split(key_hash, c["seed"], c["split_fractions"])
    sets = [set(key_hash[splits == i]) for i in range(3)]
    intersections = {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in ((0, 1), (0, 2), (1, 2))}
    assert not any(intersections.values())
    train = np.flatnonzero(splits == 0)
    # One eligible representative per equivalence group; never a val/test candidate.
    representative = {}
    for i in train[np.argsort(ids[train], kind="stable")]:
        representative.setdefault(key_hash[i], int(i))
    strata = defaultdict(list)
    for i in representative.values():
        r = rows[i]
        glycan = int(bool(r["sugar_ring_count"] or r["inositol_ring_count"]))
        group = (min(int(r["P_count"]), 2), min(int(r["core_average_mass_Da"] // 100), 8), glycan)
        strata[group].append(i)
    groups = sorted(strata)
    quota = balanced_quotas([len(strata[g]) for g in groups], c["template_pool_size"])
    chosen = []
    stratum_report = []
    for group, take in zip(groups, quota):
        candidates = np.asarray(strata[group])
        vectors = [DataStructs.CreateFromBinaryText(bytes(packed[i])) for i in candidates]
        local = MaxMinPicker().LazyBitVectorPick(vectors, len(vectors), int(take), seed=c["seed"])
        chosen.extend(candidates[list(local)].tolist())
        stratum_report.append(
            {
                "P_category": group[0],
                "mass_100Da_bin": group[1],
                "sugar_or_inositol": group[2],
                "TRAIN_candidates": len(candidates),
                "selected": int(take),
            }
        )
        emit(
            stage="TRAIN_only_stratified_MaxMin",
            stratum=list(group),
            selected=len(chosen),
            total=c["template_pool_size"],
        )
    bank_rows = np.asarray(sorted(chosen, key=lambda i: ids[i]), dtype=np.int32)
    assert len(bank_rows) == c["template_pool_size"] and np.all(splits[bank_rows] == 0)
    assert len(set(key_hash[bank_rows])) == len(bank_rows)
    refs, sim, audit = exact_retrieve(
        packed,
        packed[bank_rows],
        key_hash,
        key_hash[bank_rows],
        ids[bank_rows],
        k=c["K"],
        batch_size=c["retrieval"]["batch_size"],
        candidate_count=c["retrieval"]["candidate_count"],
        audit_count=c["retrieval"]["oracle_sample"],
        event=emit,
    )
    valid = (refs >= 0) & (sim >= c["minimum_similarity"])
    # Preserve K>=1 without covert Formula-only targets. Record actual low-similarity
    # fallbacks explicitly; never copy the target back into its own references.
    fallback = ~valid.any(axis=1)
    valid[fallback, 0] = refs[fallback, 0] >= 0
    refs[~valid] = -1
    assert (refs >= 0).any(axis=1).all()
    assert not np.any((key_hash[bank_rows[np.maximum(refs, 0)]] == key_hash[:, None]) & (refs >= 0))
    bank_target = np.zeros(len(rows), dtype=bool)
    bank_target[bank_rows] = True
    assert bank_target.sum() == 8000 and np.all(splits[bank_target] == 0)
    fitted = FormulaStandardizer.fit(
        {
            "template_formulas": torch.tensor(
                formulas[bank_rows[np.maximum(refs[part], 0)]].astype(np.float32)
            ),
            "delta_formulas": torch.tensor(
                (formulas[part, None] - formulas[bank_rows[np.maximum(refs[part], 0)]]).astype(
                    np.float32
                )
            ),
            "template_mask": torch.tensor(refs[part] >= 0),
        }
        for part in np.array_split(train, max(1, len(train) // 2048))
    )
    write(folder / "formula_statistics.json", {**fitted, "fit_split": "TRAIN_only"})
    arrays = {
        "tokens": tokens,
        "token_lengths": lengths,
        "splits": splits,
        "formulas": formulas,
        "packed": packed,
        "functional_masks": masks,
        "equivalence_keys": key_hash,
        "bank_rows": bank_rows,
        "bank_packed": packed[bank_rows],
        "references": refs,
        "similarities": sim,
        "bank_target_mask": bank_target,
        "low_similarity_fallback": fallback,
    }
    for name, array in arrays.items():
        np.save(folder / (name + ".npy"), array)
    private = table.append_column(
        "split", pa.array([("train", "validation", "test")[i] for i in splits])
    )
    private = private.append_column("is_template_bank_member", pa.array(bank_target))
    private = private.append_column("split_equivalence_key", pa.array(keys))
    pq.write_table(private, folder / "targets_private.parquet", compression="zstd")
    pq.write_table(
        private.take(pa.array(bank_rows)),
        folder / "template_bank_private.parquet",
        compression="zstd",
    )
    rng = np.random.default_rng(20261026)
    for name, eligible, number in (
        ("validation512", np.flatnonzero(splits == 1), 512),
        ("test512", np.flatnonzero(splits == 2), 512),
        ("bank_targets512", bank_rows, 512),
    ):
        np.save(folder / (name + ".npy"), np.sort(rng.choice(eligible, number, replace=False)))
    coverage = {}
    for name, eligible in (
        ("train", train),
        ("validation", np.flatnonzero(splits == 1)),
        ("test", np.flatnonzero(splits == 2)),
        ("bank_targets_LOO", bank_rows),
    ):
        best = sim[eligible, 0]
        coverage[name] = {
            "head_Morgan_Tanimoto": summary(best),
            "low_similarity_fallback_targets": int(fallback[eligible].sum()),
            "best_template_formula_L1": summary(
                np.abs(
                    formulas[eligible] - formulas[bank_rows[np.maximum(refs[eligible, 0], 0)]]
                ).sum(axis=1)
            ),
        }
    bank_masks = masks[bank_rows]
    coverage["functional_category_coverage"] = {
        str(bit): {
            "all_targets": int(((masks >> bit) & 1).sum()),
            "templates": int(((bank_masks >> bit) & 1).sum()),
        }
        for bit in range(16)
    }
    split_counts = {
        name: int((splits == i).sum()) for i, name in enumerate(("train", "validation", "test"))
    }
    audit.update(
        status="passed",
        split_counts=split_counts,
        equivalence_groups=len(set(key_hash)),
        equivalent_extra_role_records=len(rows) - len(set(key_hash)),
        cross_split_equivalence_intersections=intersections,
        bank_entries_retained_as_prediction_targets=int(bank_target.sum()),
        bank_self_or_equivalent_references=0,
        bank_val_test_entries=0,
        stratum_selection=stratum_report,
        token_length_distribution=summary(lengths),
        token_limit=c["max_label_tokens"],
        truncated_or_dropped_labels=0,
        minimum_similarity=c["minimum_similarity"],
        below_threshold_nearest_fallbacks=int(fallback.sum()),
        formula_basis=c["formula_scope"],
        all_frame_root_formula_checks=frame_checks,
        coverage=coverage,
        retrieval_uses_target_structure="oracle structural pretraining; not MS/MS inference",
        inherited_library_proofs=(
            "all source main cores have H-inclusive assembly proof; "
            "no old exclusion filters reapplied"
        ),
    )
    write(folder / "data_audit.json", audit)
    asset_names = [p.name for p in folder.iterdir() if p.is_file() and p.name != "manifest.json"]
    steps = math.ceil(split_counts["train"] / c["batch_size"])
    manifest = {
        "status": "passed",
        "schema": c["schema"],
        "configuration_sha256": digest(CONFIG),
        "inputs_sha256": {
            str(source): digest(source),
            str(source.parent / "library_manifest_v04.json"): digest(
                source.parent / "library_manifest_v04.json"
            ),
        },
        "assets_sha256": {name: digest(folder / name) for name in sorted(asset_names)},
        "targets": len(rows),
        "P_targets": int(sum(r["contains_P"] for r in rows)),
        "nonP_targets": int(sum(not r["contains_P"] for r in rows)),
        "split_counts": split_counts,
        "steps_per_epoch": steps,
        "max_optimizer_steps": steps * c["max_epochs"],
        "bank_size": len(bank_rows),
        "bank_also_prediction_targets": True,
        "bank_target_self_equivalent_masked": True,
        "bank_is_train_only": True,
        "max_observed_label_tokens": int(lengths.max()),
        "vocabulary_size": len(tok),
        "elapsed_seconds": time.monotonic() - began,
    }
    write(folder / "manifest.json", manifest)
    verify_data(root)
    emit(
        stage="data_freeze_passed",
        **{
            k: manifest[k]
            for k in (
                "targets",
                "split_counts",
                "bank_size",
                "steps_per_epoch",
                "max_optimizer_steps",
                "elapsed_seconds",
            )
        },
    )


if __name__ == "__main__":
    main()
