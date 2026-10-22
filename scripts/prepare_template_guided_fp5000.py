"""CPU-only nested template pools and independent fixed-5k training arrays."""

import argparse
import heapq
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from rdkit import Chem, DataStructs

from structure_pretrain.data.prepare_v03 import atom_counts
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.library_builder.validation import validate_polar_fragment
from structure_pretrain.polar_only_v04.chemistry import chainless_molecule, derive_polar_formula
from structure_pretrain.polar_only_v04.features import fragment_molecule, packed_fingerprint
from structure_pretrain.reported_headgroups.chemistry import builder_trial
from structure_pretrain.reported_headgroups.formal_template_data import (
    CONFIG,
    FEATURES,
    ROOT,
    V03,
    config,
    digest,
    read,
    write,
)
from structure_pretrain.reported_headgroups.template_selection import (
    MORGAN,
    coverage,
    phosphate_comparison_key,
    read_csv,
    write_csv,
)
from structure_pretrain.retrieval.indexed_morgan import make_index, search_many
from structure_pretrain.utils.formula import FormulaStandardizer


def emit(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def stratum(row):
    mass = int(row["average_mass_Da"] // 100)
    halogen = any(row[e] for e in ("F", "Cl", "Br", "I"))
    return f"P{row['P']}/MW{mass}/N{bool(row['N'])}/S{bool(row['S'])}/X{halogen}"


def choose_nested(
    candidates, candidate_vectors, candidate_fps, descriptions, seeds, seed_fps, count
):
    """Density-first sphere exclusion in population-balanced chemistry strata."""
    index = make_index(candidate_vectors, expansion_search=128, threads=4)
    density = np.zeros(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), 4096):
        hits = search_many(index, candidate_vectors[start : start + 4096], count=6, threads=4)
        density[start : start + len(hits.counts)] = 1 - hits.distances[:, 5]
    del index
    buckets = defaultdict(list)
    for local, group in enumerate(candidates):
        buckets[stratum(descriptions[group])].append(local)
    for values in buckets.values():
        values.sort(key=lambda j: (-float(density[j]), candidates[j]))
    weights = {k: len(v) ** 0.5 for k, v in buckets.items()}
    positions = Counter()
    chosen_count = Counter()
    heap = [(0.0, key) for key in sorted(buckets)]
    heapq.heapify(heap)
    selected, fingerprints = list(seeds), list(seed_fps)
    selected_keys = {r["comparison_key"] for r in seeds}
    rejected = 0
    while len(selected) < count and heap:
        _, key = heapq.heappop(heap)
        values = buckets[key]
        accepted = False
        while positions[key] < len(values):
            local = values[positions[key]]
            positions[key] += 1
            group = candidates[local]
            candidate = descriptions[group]
            if candidate["comparison_key"] in selected_keys:
                continue
            maximum = max(
                DataStructs.BulkTanimotoSimilarity(candidate_fps[local], fingerprints), default=0
            )
            if maximum > 0.85:
                rejected += 1
                continue
            selected.append(candidate)
            fingerprints.append(candidate_fps[local])
            selected_keys.add(candidate["comparison_key"])
            chosen_count[key] += 1
            accepted = True
            break
        if positions[key] < len(values):
            heapq.heappush(heap, (chosen_count[key] / weights[key], key))
        if accepted and len(selected) % 1000 == 0:
            emit(
                stage="nested_template_selection",
                selected=len(selected),
                ceiling=count,
                near_duplicate_skips=rejected,
            )
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} distinct templates; never fill by duplicate graphs")
    return selected, {
        "method": "density-first sphere exclusion, mass/P/N/S/halogen strata, sqrt population allocation",
        "density": "approximate sixth-neighbour Tanimoto in TRAIN-only HNSW",
        "new_to_previous_similarity_ceiling": 0.85,
        "near_duplicate_skips": rejected,
        "strata": len(buckets),
        "all_selection_targets_train_only": True,
    }


def retrieve(queries, bank_vectors, query_keys, bank_keys, k=10):
    index = make_index(bank_vectors, expansion_search=256, threads=4)
    bank_fp = [DataStructs.CreateFromBinaryText(bytes(r)) for r in bank_vectors]
    references = np.full((len(queries), k), -1, dtype=np.int32)
    scores = np.zeros((len(queries), k), dtype=np.float32)
    bank_keys = np.asarray(bank_keys)
    # Approximate candidates, exact 2048-bit reranking, then explicit own-core exclusion.
    for start in range(0, len(queries), 4096):
        hits = search_many(
            index, queries[start : start + 4096], count=min(64, len(bank_vectors)), threads=4
        )
        for offset in range(len(hits.counts)):
            i = start + offset
            choices = [
                int(x)
                for x in hits.keys[offset, : hits.counts[offset]]
                if bank_keys[int(x)] != query_keys[i]
            ]
            fingerprint = DataStructs.CreateFromBinaryText(bytes(queries[i]))
            sims = DataStructs.BulkTanimotoSimilarity(fingerprint, [bank_fp[x] for x in choices])
            ordered = sorted(zip(choices, sims), key=lambda x: (-x[1], x[0]))[:k]
            for j, (choice, sim) in enumerate(ordered):
                references[i, j], scores[i, j] = choice, sim
        if start % 20000 < 4096:
            emit(
                stage="reference_cache",
                targets=min(start + 4096, len(queries)),
                total=len(queries),
                bank=len(bank_vectors),
            )
    return (
        references,
        scores,
        exact_top1_audit(queries, bank_vectors, query_keys, bank_keys, scores),
    )


def exact_top1_audit(queries, bank_vectors, query_keys, bank_keys, scores):
    bank_fp = [DataStructs.CreateFromBinaryText(bytes(r)) for r in bank_vectors]
    bank_keys = np.asarray(bank_keys)
    sample = np.random.default_rng(817).choice(len(queries), min(64, len(queries)), replace=False)
    matched = 0
    for i in sample:
        exact = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                DataStructs.CreateFromBinaryText(bytes(queries[i])), bank_fp
            )
        )
        exact[bank_keys == query_keys[i]] = -1
        matched += abs(float(exact.max()) - float(scores[i, 0])) < 1e-6
    if matched / len(sample) < 0.95:
        raise ValueError(f"Retrieval exact top1 audit failed: {matched}/{len(sample)}")
    return {
        "exact_top1_score_matches": matched,
        "exact_audit_targets": len(sample),
        "fraction": matched / len(sample),
        "candidate_count": 64,
        "expansion_search": 256,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("outputs").resolve()) or root.name != ROOT.name:
        raise ValueError("Independent dedicated formal output root required")
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    if (data / "manifest.json").exists():
        from structure_pretrain.reported_headgroups.formal_template_data import verify_data

        emit(stage="reuse_verified_prepared_data", manifest=verify_data(root))
        return
    began = time.monotonic()
    specification = config()
    library_path = Path("data/structure_library_v0.3/polar_library.parquet")
    descriptor_path = Path(
        "outputs/polar_size_scope_audit_20261022/per_graph_descriptors_private.parquet"
    )
    prototype = Path("outputs/headgroup_template_proposal_20261022")
    immutable = [
        library_path,
        descriptor_path,
        V03 / "polar_groups.parquet",
        V03 / "polar_groups.npy",
        V03 / "row_splits.npy",
        V03 / "chain_labels.npy",
        V03 / "formulas.npy",
        V03 / "polar_tokenizer.json",
        FEATURES / "formula_audit.json",
        FEATURES / "features_manifest.json",
        FEATURES / "head_polar_formulas.npy",
        FEATURES / "chainless_fingerprints.npy",
        FEATURES / "variant_heads.npy",
        FEATURES / "variant_occupancy.npy",
        FEATURES / "chain_counts.npy",
        FEATURES / "scaffold_counts.npy",
        prototype / "approved_reference_catalog.csv",
        prototype / "selection_cache_private.npz",
    ]
    frozen = {str(p): digest(p) for p in immutable}
    assert read(FEATURES / "formula_audit.json")["status"] == "passed"
    library = pq.read_table(library_path).to_pylist()
    heads = pq.read_table(V03 / "polar_groups.parquet").to_pylist()
    assert [r["group"] for r in heads] == list(range(len(heads)))
    by_smiles = {r["canonical_polar_smiles"]: r["group"] for r in heads}
    descriptor = {r["polar_id"]: r for r in pq.read_table(descriptor_path).to_pylist()}
    with np.load(prototype / "selection_cache_private.npz", allow_pickle=False) as cached:
        cached_packed, cached_keys = cached["packed"], cached["keys"]
    groups = [None] * len(heads)
    root_vectors = np.zeros((len(heads), 256), dtype=np.uint8)
    keys = np.empty(len(heads), dtype=object)
    for i, row in enumerate(library):
        g = by_smiles[row["polar_fragment_smiles"]]
        groups[g] = {
            **descriptor[row["polar_id"]],
            "frozen_group": g,
            "template_id": row["polar_id"],
            "source": row["source"],
            "structure_origin": row["structure_origin"],
            "polar_smiles": row["polar_fragment_smiles"],
            "comparison_key": str(cached_keys[i]),
            "chemistry_family": row["chemistry_family"],
            "source_cids": row["source_cids"],
        }
        root_vectors[g] = cached_packed[i]
        keys[g] = cached_keys[i]
    split = np.asarray(
        [{"train": 0, "validation": 1, "test": 2}[r["split"]] for r in heads], dtype=np.uint8
    )
    heldout_keys = set(keys[split != 0])
    test_keys = set(keys[split == 2])
    scope = np.asarray(
        [
            not r["extra_anchor"] and r["average_mass_Da"] <= 600 and r["smiles_tokens"] + 2 <= 84
            for r in groups
        ]
    )
    overlap = (split == 0) & np.asarray([key in heldout_keys for key in keys])
    overlap |= (split == 1) & np.asarray([key in test_keys for key in keys])
    scope &= ~overlap
    train_candidates = np.flatnonzero(scope & (split == 0)).tolist()
    source_by_key = {groups[g]["comparison_key"]: g for g in train_candidates}
    all_keys = set(keys)
    seeds, seed_fps, seed_keys = [], [], set()
    for known in sorted(
        read_csv(prototype / "approved_reference_catalog.csv"), key=lambda r: r["headgroup_id"]
    ):
        if known["builder_compatible"] != "True":
            continue
        key = phosphate_comparison_key(known["polar_smiles"])
        if key in heldout_keys or key in seed_keys:
            continue
        if key in source_by_key:
            row = groups[source_by_key[key]]
        elif key not in all_keys:
            validate_polar_fragment(known["polar_smiles"], 1)
            row = {
                "frozen_group": -1,
                "template_id": known["headgroup_id"],
                "polar_smiles": known["polar_smiles"],
                "comparison_key": key,
                "source": known["source"],
                "structure_origin": "external_human_approved_prior",
                "chemistry_family": "known_reference",
                "average_mass_Da": float(known["average_mass_Da"]),
                "source_cids": [],
                "P": int(known["P"]),
                "C": int(known["C"]),
                "N": int(known["N"]),
                "S": int(known["S"]),
                "F": int(known["F"]),
                "Cl": int(known["Cl"]),
                "Br": int(known["Br"]),
                "I": int(known["I"]),
            }
        else:
            continue
        row = dict(row)
        row["approved_seed_id"] = known["headgroup_id"]
        seeds.append(row)
        seed_fps.append(MORGAN.GetFingerprint(Chem.MolFromSmiles(row["polar_smiles"])))
        seed_keys.add(key)
    emit(
        stage="formal_scope",
        frozen_heads=len(heads),
        eligible_train=len(train_candidates),
        approved_seeds=len(seeds),
        split_counts=dict(Counter(int(x) for x in split[scope])),
        normalized_cross_split_exclusions=int(overlap.sum()),
    )
    candidate_fps = [
        DataStructs.CreateFromBinaryText(bytes(root_vectors[g])) for g in train_candidates
    ]
    nested_path = data / "nested_templates_20000_private.json"
    if nested_path.exists():
        selected = read(nested_path)
        selection = read(data / "template_selection.json")
        assert selection["frozen_input_sha256"] == frozen, "Partial selection inputs changed"
        assert selection["configuration_sha256"] == digest(CONFIG), (
            "Partial selection configuration changed"
        )
    else:
        selected, selection = choose_nested(
            train_candidates,
            root_vectors[train_candidates],
            candidate_fps,
            groups,
            seeds,
            seed_fps,
            20000,
        )
        selection["frozen_input_sha256"] = frozen
        selection["configuration_sha256"] = digest(CONFIG)
        write(nested_path, selected)
        write(data / "template_selection.json", selection)
    assert len({r["comparison_key"] for r in selected}) == 20000
    assert all(r["comparison_key"] not in heldout_keys for r in selected)
    for size in specification["template_pool_sizes"]:
        write_csv(
            data / f"templates_{size}_private.csv",
            selected[:size],
            [
                "template_id",
                "frozen_group",
                "polar_smiles",
                "comparison_key",
                "source",
                "structure_origin",
                "chemistry_family",
                "average_mass_Da",
                "approved_seed_id",
                "source_cids",
            ],
        )
    primary = selected[:5000]
    primary_keys = [r["comparison_key"] for r in primary]
    primary_packed = np.empty((2, 5000, 256), dtype=np.uint8)
    primary_formulas = np.empty((5000, 10), dtype=np.int16)
    for i, row in enumerate(primary):
        mol = validate_polar_fragment(row["polar_smiles"], 1)
        primary_formulas[i] = atom_counts(mol)
        trial = builder_trial(row["polar_smiles"])
        assert trial["independent_element_H_balance"]
        for occupancy, chain_count in enumerate((2, 1)):
            fragment = chainless_molecule(row["polar_smiles"], chain_count)
            fragment_molecule(Chem.MolToSmiles(fragment, isomericSmiles=False))
            primary_packed[occupancy, i] = packed_fingerprint(fragment)
        if (i + 1) % 1000 == 0:
            emit(stage="primary_builder_formula_feature_audit", templates=i + 1)
    np.save(data / "bank_packed.npy", primary_packed)
    np.save(data / "bank_formulas.npy", primary_formulas)
    frozen_packed = np.load(FEATURES / "chainless_fingerprints.npy", mmap_mode="r")
    variant_heads = np.load(FEATURES / "variant_heads.npy")
    variant_occupancy = np.load(FEATURES / "variant_occupancy.npy")
    lookup = np.full((len(heads), 2), -1, dtype=np.int32)
    lookup[variant_heads, 2 - variant_occupancy] = np.arange(len(variant_heads))
    assert np.all(lookup >= 0)
    retrieval_audits = []
    if (data / "references.npy").exists() and (data / "reference_similarities.npy").exists():
        # Nested selection binds every immutable input and configuration; independently
        # rebuilt bank above plus a fresh oracle audit verifies the CPU partial cache.
        references = np.load(data / "references.npy")
        similarities = np.load(data / "reference_similarities.npy")
        assert references.shape == similarities.shape == (len(heads), 2, 10)
        assert ((references >= -1) & (references < 5000)).all()
        assert (np.diff(similarities, axis=-1) <= 1e-6).all()
        for occupancy in (0, 1):
            retrieval_audits.append(
                exact_top1_audit(
                    np.asarray(frozen_packed[lookup[:, occupancy]]),
                    primary_packed[occupancy],
                    keys,
                    primary_keys,
                    similarities[:, occupancy],
                )
            )
        emit(stage="reused_partial_CPU_reference_cache_with_fresh_exact_audit")
    else:
        refs, sims = [], []
        for occupancy in (0, 1):
            indices, scores, audit = retrieve(
                np.asarray(frozen_packed[lookup[:, occupancy]]),
                primary_packed[occupancy],
                keys,
                primary_keys,
            )
            refs.append(indices)
            sims.append(scores)
            retrieval_audits.append(audit)
        references = np.stack(refs, axis=1)
        similarities = np.stack(sims, axis=1)
    # Preserve the structural threshold. Any zero-coverage core is exposed before training.
    references[similarities < specification["minimum_structural_similarity"]] = -1
    zero = (references[:, :, 0] < 0).any(axis=1) & scope
    if zero.any():
        write(
            data / "zero_reference_failure.json",
            {"groups": np.flatnonzero(zero).tolist(), "count": int(zero.sum())},
        )
        raise ValueError(
            "Targets without a legal template; do not silently omit or forge references"
        )
    for occupancy in (0, 1):
        for start in range(0, len(heads), 20000):
            block = references[start : start + 20000, occupancy]
            valid = block >= 0
            forbidden = (
                np.asarray(primary_keys, dtype=object)[block.clip(min=0)]
                == keys[start : start + len(block), None]
            )
            assert not (valid & forbidden).any()
    np.save(data / "references.npy", references)
    np.save(data / "reference_similarities.npy", similarities)
    all_groups = np.load(V03 / "polar_groups.npy", mmap_mode="r")
    order = np.random.default_rng(42).permutation(len(all_groups))
    unique, first = np.unique(all_groups[order], return_index=True)
    assert np.array_equal(unique, np.arange(len(heads)))
    real_rows = order[first]
    chain_labels = np.load(V03 / "chain_labels.npy", mmap_mode="r")[real_rows]
    formulas = np.load(V03 / "formulas.npy", mmap_mode="r")[real_rows]
    polar = np.load(FEATURES / "head_polar_formulas.npy")
    np.testing.assert_array_equal(
        derive_polar_formula(
            formulas,
            chain_labels,
            np.load(FEATURES / "chain_counts.npy"),
            np.load(FEATURES / "scaffold_counts.npy"),
        ),
        polar,
    )
    tokenizer = SmilesTokenizer(read(V03 / "polar_tokenizer.json")["tokens"])
    tokens = np.full((len(heads), 84), tokenizer.pad_id, dtype=np.int16)
    lengths = np.empty(len(heads), dtype=np.int16)
    for g, row in enumerate(heads):
        ids = tokenizer.encode(row["canonical_polar_smiles"])
        assert len(ids) <= 84
        tokens[g, : len(ids)] = ids
        lengths[g] = len(ids)
    training_split = split.copy()
    training_split[~scope] = 255
    arrays = {
        "tokens": tokens,
        "token_lengths": lengths,
        "splits": training_split,
        "target_rows": real_rows,
        "full_formulas": formulas,
        "chain_labels": chain_labels,
        "polar_formulas": polar,
        "occupancy": (chain_labels[:, 1] == 0).astype(np.uint8),
    }
    for name, value in arrays.items():
        np.save(data / (name + ".npy"), value)

    # Statistics use only TRAIN target/reference pairs for this exact bank and K.
    def examples():
        train = np.flatnonzero(training_split == 0)
        for start in range(0, len(train), 4096):
            g = train[start : start + 4096]
            r = references[g, arrays["occupancy"][g], :3]
            mask = r >= 0
            f = primary_formulas[r.clip(min=0)]
            yield {
                "template_formulas": torch.tensor(f),
                "delta_formulas": torch.tensor(polar[g, None] - f),
                "template_mask": torch.tensor(mask),
            }

    statistics = FormulaStandardizer.fit(examples())
    statistics.update({"training_pairs_only": True, "K": 3, "seed": 42, "bank_size": 5000})
    write(data / "formula_statistics.json", statistics)
    diagnostic_rows = np.load("outputs/pretraining_v04/samples/diagnostic_512.npy")
    diagnostic_groups = np.asarray(all_groups[diagnostic_rows], dtype=np.int64)
    np.save(data / "diagnostic_512_original_rows.npy", diagnostic_rows)
    np.save(data / "diagnostic_512_groups.npy", diagnostic_groups)
    write(
        data / "diagnostic_scope.json",
        {
            "frozen_original_count": 512,
            "in_training_scope": int(scope[diagnostic_groups].sum()),
            "representation": "same frozen head groups; current representative chain rows separate from original row IDs",
        },
    )
    # Rooted-Polar structural and elemental coverage, all frozen cores, after selection only.
    bank_root = np.asarray(
        [
            np.frombuffer(
                DataStructs.BitVectToBinaryText(
                    MORGAN.GetFingerprint(Chem.MolFromSmiles(r["polar_smiles"]))
                ),
                dtype=np.uint8,
            )
            for r in selected
        ]
    )
    comparison = {}
    from scipy.spatial import cKDTree

    all_counts = np.asarray(polar, dtype=np.float64)
    bank_counts = np.asarray(
        [atom_counts(Chem.MolFromSmiles(r["polar_smiles"])) for r in selected], dtype=np.float64
    )
    for size in specification["template_pool_sizes"]:
        idx, score, audit = retrieve(
            root_vectors,
            bank_root[:size],
            keys,
            [r["comparison_key"] for r in selected[:size]],
            k=1,
        )
        tree = cKDTree(bank_counts[:size])
        distances, positions = tree.query(all_counts, k=2, p=1, workers=4)
        own = np.asarray(
            [selected[int(j)]["comparison_key"] == key for j, key in zip(positions[:, 0], keys)]
        )
        formula_distance = np.where(own, distances[:, 1], distances[:, 0])
        comparison[str(size)] = {
            "all_frozen_cores": coverage(score[:, 0]),
            "retrieval_exact_audit": audit,
            "minimum_formula_L1_including_H_quantiles": {
                str(q): float(np.quantile(formula_distance, q)) for q in (0, 0.1, 0.5, 0.9, 1)
            },
            "formula_exact_different_core_fraction": float(np.mean(formula_distance == 0)),
            "coverage_role": "post-selection diagnosis only; not checkpoint or template tuning on test",
        }
        emit(stage="template_scale_coverage", size=size, coverage=coverage(score[:, 0]))
    write(data / "template_scale_coverage.json", comparison)
    for path, value in frozen.items():
        assert digest(path) == value, "Frozen source changed"
    assets = {p.name: digest(p) for p in sorted(data.glob("*.npy"))}
    for name in (
        "formula_statistics.json",
        "nested_templates_20000_private.json",
        "template_selection.json",
    ):
        assets[name] = digest(data / name)
    manifest = {
        "status": "passed",
        "schema": specification["schema"],
        "frozen_input_sha256": frozen,
        "data_assets_sha256": assets,
        "frozen_core_count": len(heads),
        "scope_core_count": int(scope.sum()),
        "split_counts": {
            name: int((training_split == i).sum())
            for i, name in enumerate(("train", "validation", "test"))
        },
        "one_real_target_row_per_core": True,
        "split_origin": "unchanged frozen graph-level 90/5/5; derived scope removes anchors and normalized overlap only",
        "seed_templates": len(seeds),
        "main_bank_size": 5000,
        "nested_sizes": specification["template_pool_sizes"],
        "retrieval_audits": retrieval_audits,
        "primary5000_builder_independent_H_formula_checks": 5000,
        "no_target_core_self_reference": True,
        "no_validation_or_test_reference_graphs": True,
        "formal_selection_rule": "minimum_full_validation_token_CE; patience5 complete epochs",
        "selection": selection,
        "elapsed_seconds": time.monotonic() - began,
    }
    write(data / "manifest.json", manifest)
    emit(stage="prepared", manifest=manifest)


if __name__ == "__main__":
    main()
