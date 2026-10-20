"""Independent, graph-diverse TRAIN pool and immutable FP50k contracts."""

import argparse
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.prepare import LIBRARY, SAMPLES, V03, dump, row_hash
from structure_pretrain.polar_only_v04.retrieval import chosen_mode, exact_scores, quantiles
from structure_pretrain.polar_only_v04.runtime import load_config

ROOT = Path("outputs/polar_fp_base_50k_20261020")
FEATURES = Path("data/private/pretraining_v04")
CONFIG = Path("configs/polar_fp_base_50k.yaml")
SCHEMA = "polar_fp_base_50k_v1"


def fresh_directory(path, root=ROOT):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not root.is_relative_to(Path("outputs").resolve()) or not path.is_relative_to(root):
        raise ValueError("Independent FP50k output directory required")
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    return path


def input_paths():
    return (
        [
            V03 / name
            for name in (
                "labels_manifest.json",
                "polar_tokenizer.json",
                "chain_vocab.json",
                "row_splits.npy",
                "polar_groups.npy",
                "polar_groups.parquet",
            )
        ]
        + [
            FEATURES / name
            for name in (
                "features_manifest.json",
                "retrieval_audit.json",
                "formula_stats_train_only.json",
                "H_rows.npy",
                "HC_rows.npy",
                "H_scores.npy",
                "HC_scores.npy",
                "row_variants.npy",
                "head_polar_formulas.npy",
                "chain_counts.npy",
                "scaffold_counts.npy",
                "chainless_fingerprints.npy",
            )
        ]
        + [LIBRARY / "polar_library.parquet"]
    )


def source_hashes():
    paths = list(Path("src/structure_pretrain/polar_only_v04").glob("*.py"))
    paths += list(Path("src/structure_pretrain/models").glob("*.py"))
    paths += [
        Path(p)
        for p in (
            "train_polar_fp50k.py",
            "train_structure_real.py",
            "configs/polar_fp_base_50k.yaml",
            "scripts/supervise_polar_fp50k.py",
            "scripts/supervise_formula_beam_cuda_v04.py",
            "src/structure_pretrain/training/quality_v03.py",
            "src/structure_pretrain/utils/attention.py",
        )
    ]
    return {p.as_posix(): file_hash(p) for p in sorted(paths)}


def get_contract(root=ROOT, verify_assets=False):
    manifest = read_json(Path(root) / "pool" / "manifest_private.json")
    rows = np.load(Path(root) / "pool" / "train_pool_50k.npy")
    if len(rows) != 50000 or row_hash(rows) != manifest["target_rows_sha256"]:
        raise ValueError("FP50k pool hash/count mismatch")
    if verify_assets:
        for name, expected in manifest["inputs_sha256"].items():
            if file_hash(Path(name)) != expected:
                raise ValueError(f"Frozen input changed: {name}")
    config = load_config(CONFIG)
    baseline = asdict(load_config("configs/polar_only_v04_fp.yaml"))
    baseline["train"]["checkpoint_every"] = 1000
    if asdict(config) != baseline:
        raise ValueError("FP50k must freeze the existing FP architecture and optimizer settings")
    contract = {
        "schema": SCHEMA,
        "config": asdict(config),
        "pool_manifest_sha256": file_hash(Path(root) / "pool" / "manifest_private.json"),
        "target_rows_sha256": row_hash(rows),
        "inputs_sha256": manifest["inputs_sha256"],
        "source_sha256": source_hashes(),
        "initialization": "random_seed_42_no_pretrained_weights",
        "precision": "fp32",
        "attention": "math",
        "tf32": False,
        "optimizer": "AdamW",
        "outputs": ["polar_SMILES_only"],
    }
    return config, rows, contract


def distribution(rows, groups, chains, heads, formulas, vocab):
    gs = groups[rows]
    return {
        "rows": len(rows),
        "distinct_polar_graphs": len(np.unique(gs)),
        "distinct_polar_formulas": len(np.unique(formulas[gs], axis=0)),
        "family": dict(sorted(Counter(heads["chemistry_family"][int(g)] for g in gs).items())),
        "length_bins": dict(
            sorted(Counter(str(heads["token_length"][int(g)] // 16 * 16) for g in gs).items())
        ),
        "token_length_quantiles": quantiles([heads["token_length"][int(g)] for g in gs]),
        "chain_count": dict(Counter(str(int((chains[r] != 0).sum())) for r in rows)),
        "linkage_combinations": dict(
            Counter(
                "/".join(
                    "O" if vocab[t].startswith("O-") else "P" if vocab[t].startswith("P-") else "A"
                    for t in chains[r]
                    if t
                )
                for r in rows
            )
        ),
    }


def prepare(root=ROOT):
    out = fresh_directory(Path(root) / "pool", root)
    splits = np.load(V03 / "row_splits.npy", mmap_mode="r")
    groups = np.load(V03 / "polar_groups.npy", mmap_mode="r")
    chains = np.load(V03 / "chain_labels.npy", mmap_mode="r")
    formulas = np.load(FEATURES / "head_polar_formulas.npy", mmap_mode="r")
    heads = pq.read_table(V03 / "polar_groups.parquet").to_pydict()
    vocab = read_json(V03 / "chain_vocab.json")["tokens"]
    train = np.flatnonzero(splits == 0)
    rng = np.random.default_rng(42)
    # A random real lipid per TRAIN graph; no FA-swap duplication of graph targets.
    randomized = rng.permutation(train)
    _, first = np.unique(groups[randomized], return_index=True)
    representatives = randomized[np.sort(first)]
    # Retain real rare vinyl-ether rows and named special cases when present in TRAIN.
    p_tokens = np.asarray([i for i, token in enumerate(vocab) if token.startswith("P-")])
    rare = randomized[np.isin(chains[randomized], p_tokens).any(1)]
    rare_rows = {}
    for r in rare:
        rare_rows.setdefault(int(groups[r]), int(r))
    representatives = np.asarray(
        [rare_rows.get(int(groups[r]), int(r)) for r in representatives], dtype=np.int64
    )
    polar = pq.read_table(
        LIBRARY / "polar_library.parquet", columns=["polar_fragment_smiles", "is_special_case"]
    ).to_pydict()
    special = {
        s for s, flag in zip(polar["polar_fragment_smiles"], polar["is_special_case"]) if flag
    }
    reserved = [
        int(r)
        for r in representatives
        if int(groups[r]) in rare_rows or heads["canonical_polar_smiles"][int(groups[r])] in special
    ]
    reserved_groups = {int(groups[r]) for r in reserved}
    buckets = defaultdict(lambda: defaultdict(list))
    for r in representatives:
        g = int(groups[r])
        if g in reserved_groups:
            continue
        key = (
            heads["chemistry_family"][g],
            heads["token_length"][g] // 16,
            tuple(np.flatnonzero(formulas[g])),
        )
        buckets[key][tuple(formulas[g])].append(int(r))
    queues = {}
    for key in sorted(buckets):
        # Within a family/length/element bucket, cover formulas before repeating one.
        fs = list(buckets[key].values())
        rng.shuffle(fs)
        q = []
        while fs:
            for values in fs:
                q.append(values.pop())
            fs = [values for values in fs if values]
        queues[key] = deque(q)
    selected = reserved.copy()
    while len(selected) < 50000:
        for key in queues:
            if queues[key]:
                selected.append(queues[key].popleft())
                if len(selected) == 50000:
                    break
        if not any(queues.values()) and len(selected) < 50000:
            raise ValueError("Insufficient distinct TRAIN graphs; no synthetic padding")
    rows = np.asarray(selected, dtype=np.int64)
    rng.shuffle(rows)
    assert np.all(splits[rows] == 0) and len(np.unique(groups[rows])) == 50000
    h = np.load(FEATURES / "H_rows.npy", mmap_mode="r")
    hc = np.load(FEATURES / "HC_rows.npy", mmap_mode="r")
    mode, fallback = chosen_mode(rows, (h[rows] >= 0).sum(1), (hc[rows] >= 0).sum(1))
    refs = np.where(mode[:, None], h[rows, :3], hc[rows, :3])
    good = refs >= 0
    target = np.broadcast_to(rows[:, None], refs.shape)[good]
    reference = refs[good]
    assert np.all(splits[reference] == 0)
    assert np.all(groups[reference] != groups[target]) and np.all(reference != target)
    same = np.all(chains[reference] == chains[target], axis=1)
    assert np.array_equal(same, np.broadcast_to(mode[:, None], refs.shape)[good])
    variants = np.load(FEATURES / "row_variants.npy", mmap_mode="r")
    fp = np.load(FEATURES / "chainless_fingerprints.npy", mmap_mode="r")
    sim = exact_scores(fp[variants[target]], fp[variants[reference]])
    assert np.all(sim >= 0.2)
    audit = read_json(FEATURES / "retrieval_audit.json")
    stats = read_json(FEATURES / "formula_stats_train_only.json")
    assert stats["training_pairs_only"] and stats["retrieval_audit_sha256"] == file_hash(
        FEATURES / "retrieval_audit.json"
    )
    assert (stats["K"], stats["h_fraction"], stats["seed"]) == (3, 0.8, 42)
    for name, expected in audit["artifacts_sha256"].items():
        assert file_hash(FEATURES / name) == expected
    np.save(out / "train_pool_50k.npy", rows)
    old = np.load(SAMPLES / "train_pool.npy")
    report = {
        "schema": SCHEMA,
        "seed": 42,
        "target_rows_sha256": row_hash(rows),
        "selection": "one real TRAIN row per graph; balanced family/length/element strata; "
        "formula round-robin within each stratum; seeded random real chain row",
        "pool": distribution(rows, groups, chains, heads, formulas, vocab),
        "old_20k_pool": distribution(old, groups, chains, heads, formulas, vocab),
        "available_train_graphs": len(representatives),
        "reserved_rare_linkage_and_special_graphs": len(reserved),
        "train_graph_distribution": distribution(
            representatives, groups, chains, heads, formulas, vocab
        ),
        "overlap_old_pool_graphs": len(np.intersect1d(groups[old], groups[rows])),
        "retrieval": {
            "K": 3,
            "threshold": 0.2,
            "H_requested": 0.8,
            "actual_H_targets": int(mode.sum()),
            "fallback_targets": int(fallback.sum()),
            "template_pairs": int(good.sum()),
            "H_pair_fraction": float(same.mean()),
            "legal_template_count_histogram": dict(Counter(map(str, good.sum(1)))),
            "similarity_quantiles": quantiles(sim),
            "all_pairs_rechecked_train_only_different_graph_chain_mode_similarity": True,
        },
        "normalizer": "reused unchanged full-TRAIN-pair K3/H0.8/seed42 statistics; "
        "same frozen retrieval contract, no validation/test fitting",
        "inputs_sha256": {p.as_posix(): file_hash(p) for p in input_paths()},
    }
    dump(out / "manifest_private.json", report)
    print(json.dumps({k: report[k] for k in ("pool", "retrieval", "target_rows_sha256")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, default=ROOT)
    prepare(p.parse_args().experiment_root)
