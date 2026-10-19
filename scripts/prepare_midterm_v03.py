"""Freeze evaluation cohorts and audit H/C/HC relationships without changing training data."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.library_builder.pipeline import file_hash

ROOT = Path("outputs/midterm_v03")
PREPARED = Path("data/private/pretraining_v03")
SIM_EDGES = np.array([0.4, 0.6, 0.8, 0.9, 1.000001])
SIM_LABELS = ["[.4,.6)", "[.6,.8)", "[.8,.9)", "[.9,1]"]


def relation_codes(target_heads, reference_heads, target_chains, reference_chains):
    same_head = target_heads == reference_heads
    same_chain = np.all(target_chains == reference_chains, axis=-1)
    # 0 = same head+chains (must be excluded under frozen unique-builder semantics).
    return np.where(
        ~same_head & same_chain,
        1,
        np.where(same_head & ~same_chain, 2, np.where(~same_head & ~same_chain, 3, 0)),
    )


def choose_stratified(eligible, strata, group_ids, count=512, seed=20261019):
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for row in rng.permutation(eligible):
        buckets[strata[int(row)]].append(int(row))
    chosen, used_heads = [], set()
    while len(chosen) < count:
        progress = False
        for key in sorted(buckets):
            while buckets[key] and int(group_ids[buckets[key][-1]]) in used_heads:
                buckets[key].pop()
            if buckets[key]:
                row = buckets[key].pop()
                chosen.append(row)
                used_heads.add(int(group_ids[row]))
                progress = True
                if len(chosen) == count:
                    break
        if not progress:
            raise ValueError("Not enough distinct validation heads for diagnostic cohort")
    return np.array(chosen, dtype=np.int64)


def audit_actual_pool(rows):
    codes = {
        m: np.array(np.load(ROOT / f"top10_relation_codes_{m}_private.npy", mmap_mode="r")[rows])
        for m in ("normal", "polar_excluded")
    }
    scores = {
        m: np.array(np.load(PREPARED / f"retrieval_{m}/scores.npy", mmap_mode="r")[rows])
        for m in codes
    }
    keys = (rows.astype(np.uint64) * 2654435761 + 42 * 2246822519) & 0xFFFFFFFF
    cross = keys / 2**32 < 0.5  # exact unchanged RealStructureDataset seed42, epoch0
    codes["effective_mixed"] = np.where(cross[:, None], codes["polar_excluded"], codes["normal"])
    scores["effective_mixed"] = np.where(cross[:, None], scores["polar_excluded"], scores["normal"])
    result = {
        "target_denominator": len(rows),
        "seed": 42,
        "epoch": 0,
        "mode_counts": {"normal": int((~cross).sum()), "polar_excluded": int(cross.sum())},
        "modes": {},
    }
    for mode, values in codes.items():
        total = int(np.count_nonzero(values))
        bitmask = np.zeros(len(rows), dtype=np.uint8)
        pairs, targets, bins = {}, {}, {}
        sim_bins = np.searchsorted(SIM_EDGES[1:], scores[mode], side="right").clip(0, 3)
        for code, label in ((1, "H"), (2, "C"), (3, "HC")):
            match = values == code
            present = match.any(1)
            bitmask |= present.astype(np.uint8) << (code - 1)
            pairs[label] = {"count": int(match.sum()), "fraction": float(match.sum() / total)}
            targets[label] = {"count": int(present.sum()), "fraction": float(present.mean())}
            bins[label] = {
                interval: {
                    "pairs": int((match & (sim_bins == b)).sum()),
                    "targets": int((match & (sim_bins == b)).any(1).sum()),
                }
                for b, interval in enumerate(SIM_LABELS)
            }
        result["modes"][mode] = {
            "pair_count": total,
            "pairs": pairs,
            "targets_with_at_least_one": targets,
            "target_type_bitmask_counts": dict(Counter(bitmask.tolist())),
            "relation_similarity_counts": bins,
        }
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "evaluation_manifest.json"
    groups = np.load(PREPARED / "polar_groups.npy", mmap_mode="r")
    splits = np.load(PREPARED / "row_splits.npy", mmap_mode="r")
    chains = np.load(PREPARED / "chain_labels.npy", mmap_mode="r")
    ids = np.load("data/private/morgan_index_v0.3/row_ids.npy", mmap_mode="r")
    head = pq.read_table(PREPARED / "polar_groups.parquet").to_pydict()
    vocab = read_json(PREPARED / "chain_vocab.json")["tokens"]
    scores = np.load(PREPARED / "retrieval_polar_excluded/scores.npy", mmap_mode="r")
    previous = Path("outputs/pretraining_v03/rocm_smoke/fingerprint_resume/sample_rows")
    train = np.load(previous / "local_smoke_train_rows.npy")
    validation = np.load(previous / "local_smoke_validation_rows.npy")
    for split, values in (("train", train), ("validation", validation)):
        other = np.load(
            Path("outputs/pretraining_v03/rocm_smoke/graph_resume/sample_rows")
            / f"local_smoke_{split}_rows.npy"
        )
        np.testing.assert_array_equal(values, other)
    cohorts = {"train_pool": train, "validation_pool": validation, "smoke_128": validation[:128]}
    strata = {}
    for row in np.flatnonzero(splits == 1):
        g = int(groups[row])
        linkage = "/".join(vocab[int(c)].split("-")[0] for c in chains[row] if c != 0)
        length = int(head["token_length"][g])
        length_bin = int(np.searchsorted([32, 48, 64], length, side="right"))
        sim_bin = min(3, int(np.searchsorted(SIM_EDGES[1:], scores[row, 0], side="right")))
        strata[int(row)] = f"{head['chemistry_family'][g]}|L{length_bin}|{linkage}|S{sim_bin}"
    eligible = np.setdiff1d(np.flatnonzero(splits == 1), cohorts["smoke_128"])
    cohorts["diagnostic_512"] = choose_stratified(eligible, strata, groups)
    metadata = {}
    for name, rows in cohorts.items():
        expected_split = 0 if name == "train_pool" else 1
        assert np.all(splits[rows] == expected_split)
        destination = ROOT / f"{name}.npy"
        if destination.exists():
            np.testing.assert_array_equal(np.load(destination), rows)
        else:
            np.save(destination, rows)
        metadata[name] = {
            "count": len(rows),
            "rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            "target_ids_sha256": hashlib.sha256(
                b"\n".join(bytes(ids[r]) for r in rows)
            ).hexdigest(),
            "canonical_head_count": len(np.unique(groups[rows])),
        }
        if expected_split == 1:
            metadata[name]["strata_counts"] = dict(
                sorted(Counter(strata[int(r)] for r in rows).items())
            )
    assert not np.intersect1d(cohorts["smoke_128"], cohorts["diagnostic_512"]).size
    labels = read_json(PREPARED / "labels_manifest.json")
    manifest = {
        "schema": "midterm-fixed-evaluation-v1",
        "seed": 20261019,
        "structures_sha256": labels["structures_sha256"],
        "labels_manifest_sha256": file_hash(PREPARED / "labels_manifest.json"),
        "split_sha256": file_hash(PREPARED / "row_splits.npy"),
        "cohorts": metadata,
        "test_targets_used": 0,
        "selection": "round-robin joint family/length/linkage/top1-similarity; 512 distinct heads",
        "smoke_128": "exact previous validation pool prefix, unchanged",
    }
    if manifest_path.exists():
        assert read_json(manifest_path) == manifest, "Frozen selection cannot be replaced"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"fixed_cohort_counts": {k: len(v) for k, v in cohorts.items()}}), flush=True)

    audit = {
        "scope": "all frozen TRAIN targets and both frozen Top10 caches",
        "target_denominator": int(np.count_nonzero(splits == 0)),
        "relation_codes": {
            "H": "different head, same chains",
            "C": "same head, different chains",
            "HC": "different head and chains",
        },
        "pair_identity_exclusion": "frozen unique canonical full graphs; self rows forbidden",
        "mixture_unchanged": {"normal": 0.5, "polar_excluded": 0.5},
        "modes": {},
    }
    for mode in ("normal", "polar_excluded"):
        refs = np.load(PREPARED / f"retrieval_{mode}/template_rows.npy", mmap_mode="r")
        sim = np.load(PREPARED / f"retrieval_{mode}/scores.npy", mmap_mode="r")
        pairs, targets, combinations, pair_bins = Counter(), Counter(), Counter(), Counter()
        target_bins = Counter()
        code_path = ROOT / f"top10_relation_codes_{mode}_private.npy"
        stored_codes = np.lib.format.open_memmap(
            code_path, mode="w+", dtype=np.uint8, shape=refs.shape
        )
        stored_codes[:] = 0  # zero means absent or outside TRAIN; frozen split is binding
        zero = 0
        for start in range(0, len(splits), 8192):
            row = np.arange(start, min(start + 8192, len(splits)))
            row = row[splits[row] == 0]
            reference = np.array(refs[row])
            valid = reference >= 0
            safe = reference.clip(min=0)
            assert not np.any((reference == row[:, None]) & valid)
            assert not np.any((splits[safe] != 0) & valid), "Held-out template reference"
            codes = relation_codes(groups[row, None], groups[safe], chains[row, None], chains[safe])
            assert not np.any((codes == 0) & valid), "Identical head+chain pair must be excluded"
            if mode == "polar_excluded":
                assert not np.any((groups[row, None] == groups[safe]) & valid)
            stored_codes[row] = np.where(valid, codes, 0)
            mask = np.zeros(len(row), dtype=np.uint8)
            bins = np.searchsorted(SIM_EDGES[1:], sim[row], side="right").clip(0, 3)
            zero += int((~valid.any(1)).sum())
            for code, label in ((1, "H"), (2, "C"), (3, "HC")):
                match = (codes == code) & valid
                pairs[label] += int(match.sum())
                present = match.any(1)
                targets[label] += int(present.sum())
                mask |= present.astype(np.uint8) << (code - 1)
                for b, interval in enumerate(SIM_LABELS):
                    pair_bins[f"{label}/{interval}"] += int((match & (bins == b)).sum())
                    target_bins[f"{label}/{interval}"] += int((match & (bins == b)).any(1).sum())
            combinations.update(mask.tolist())
        stored_codes.flush()
        total_pairs = sum(pairs.values())
        audit["modes"][mode] = {
            "pair_count": total_pairs,
            "zero_template_targets": zero,
            "pairs": {
                k: {"count": pairs[k], "fraction": pairs[k] / total_pairs} for k in ["H", "C", "HC"]
            },
            "targets_with_at_least_one": {
                k: {"count": targets[k], "fraction": targets[k] / audit["target_denominator"]}
                for k in ["H", "C", "HC"]
            },
            "target_type_bitmask_counts": dict(sorted(combinations.items())),
            "bitmask_definition": "H=1,C=2,HC=4; multi-type targets retain multiple bits",
            "relation_and_similarity_pair_counts": dict(pair_bins),
            "relation_and_similarity_target_counts": dict(target_bins),
            "similarity_target_denominator": audit["target_denominator"],
            "similarity_pair_denominator": total_pairs,
            "per_top10_codes_preserved_privately": True,
            "per_top10_codes_sha256": file_hash(code_path),
        }
        print(json.dumps({"audited_mode": mode, "pairs": total_pairs}), flush=True)
    audit["actual_20k_pool"] = audit_actual_pool(cohorts["train_pool"])
    (ROOT / "template_relation_audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
