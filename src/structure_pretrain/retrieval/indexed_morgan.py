"""Disk-backed packed Morgan fingerprints, Tanimoto HNSW, exact RDKit reranking."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import DataStructs
from usearch.index import Index

from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_plausibility.pipeline import dump_json

FAMILIES = ("phosphate", "phosphonate", "polyphosphate")
TIERS = ("CORE", "EXTENDED")
ORIGINS = (
    "pubchem_baseline",
    "pubchem_multi_attachment_split",
    "generated_phosphate_analog",
    "generated_keto_inositol",
)


def load_array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def prepare_cache(structures, output, log=print):
    structures, output = Path(structures), Path(output)
    manifest_path = output / "fingerprint_manifest.json"
    source_hash = file_hash(structures)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["structures_sha256"] != source_hash or manifest["status"] != "complete":
            raise ValueError("Fingerprint cache input changed")
        return FingerprintCache(output)
    output.mkdir(parents=True, exist_ok=True)
    source = pq.ParquetFile(structures)
    n = source.metadata.num_rows
    fp = np.lib.format.open_memmap(
        output / "fingerprints.npy", mode="w+", dtype=np.uint8, shape=(n, 256)
    )
    ids = np.lib.format.open_memmap(output / "row_ids.npy", mode="w+", dtype="S67", shape=(n,))
    labels = np.lib.format.open_memmap(
        output / "labels.npy", mode="w+", dtype=np.uint8, shape=(n, 3)
    )
    hashes = np.lib.format.open_memmap(output / "fp_hashes.npy", mode="w+", dtype="V16", shape=(n,))
    offset = 0
    for batch in source.iter_batches(
        batch_size=10000,
        columns=[
            "structure_id",
            "morgan_fp_2048",
            "chemistry_family",
            "polar_plausibility_tier",
            "structure_origin",
        ],
    ):
        rows = batch.to_pydict()
        size = len(rows["structure_id"])
        fp[offset : offset + size] = np.frombuffer(
            b"".join(rows["morgan_fp_2048"]), dtype=np.uint8
        ).reshape(size, 256)
        ids[offset : offset + size] = np.asarray(rows["structure_id"], dtype="S67")
        labels[offset : offset + size] = np.array(
            [
                [FAMILIES.index(f), TIERS.index(t), ORIGINS.index(o)]
                for f, t, o in zip(
                    rows["chemistry_family"],
                    rows["polar_plausibility_tier"],
                    rows["structure_origin"],
                )
            ],
            dtype=np.uint8,
        )
        hashes[offset : offset + size] = np.frombuffer(
            b"".join(
                hashlib.blake2b(blob, digest_size=16).digest() for blob in rows["morgan_fp_2048"]
            ),
            dtype="V16",
        )
        offset += size
        if offset % 100000 == 0:
            log(f"fingerprint cache {offset}/{n}", flush=True)
    for array in (fp, ids, labels, hashes):
        array.flush()
    unique_hashes, representatives, groups = np.unique(
        hashes, return_index=True, return_inverse=True
    )
    # Verify every grouping, so hash collisions cannot merge different fingerprints.
    for start in range(0, n, 10000):
        if not np.array_equal(
            fp[start : start + 10000], fp[representatives[groups[start : start + 10000]]]
        ):
            raise ValueError("Fingerprint digest collision; use a different exact grouping key")
    np.save(output / "group_hashes.npy", unique_hashes, allow_pickle=False)
    np.save(output / "group_of_row.npy", groups.astype(np.uint32), allow_pickle=False)
    counts = np.bincount(groups, minlength=len(unique_hashes))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    np.save(output / "posting_offsets.npy", offsets, allow_pickle=False)
    order = np.lexsort((ids, groups))
    np.save(output / "posting_rows.npy", order.astype(np.uint32), allow_pickle=False)
    vectors = np.lib.format.open_memmap(
        output / "unique_vectors.npy", mode="w+", dtype=np.uint8, shape=(len(unique_hashes), 256)
    )
    for start in range(0, len(unique_hashes), 10000):
        vectors[start : start + 10000] = fp[representatives[start : start + 10000]]
    vectors.flush()
    dump_json(
        manifest_path,
        {
            "status": "complete",
            "structures_sha256": source_hash,
            "row_count": n,
            "unique_fingerprint_count": len(unique_hashes),
            "all_hash_groups_verified_against_full_2048_bits": True,
            "radius": 2,
            "bits": 2048,
            "includeChirality": False,
            "identical_fingerprint_is_not_identical_structure": True,
            "family_labels_used_for_retrieval": False,
        },
    )
    return FingerprintCache(output)


class FingerprintCache:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads(
            (self.path / "fingerprint_manifest.json").read_text(encoding="utf-8")
        )
        self.fp = load_array(self.path / "fingerprints.npy")
        self.ids = load_array(self.path / "row_ids.npy")
        self.labels = load_array(self.path / "labels.npy")
        self.vectors = load_array(self.path / "unique_vectors.npy")
        self.group_of_row = load_array(self.path / "group_of_row.npy")
        self.offsets = load_array(self.path / "posting_offsets.npy")
        self.postings = load_array(self.path / "posting_rows.npy")

    def members(self, group, count=11):
        start, end = self.offsets[group : group + 2]
        return self.postings[start : min(end, start + count)]


class IndexedMorganRetriever:
    """Reusable full-library query API. Returns row IDs for the Parquet template loader."""

    def __init__(self, path):
        self.cache = FingerprintCache(path)
        self.index = Index.restore(self.cache.path / "morgan_groups.usearch", view=True)
        report = json.loads((self.cache.path / "exact_benchmark.json").read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise ValueError("A passing exact benchmark is required")
        self.settings = report["selected_settings"]
        self.index.expansion_search = self.settings["expansion_search"]
        self.hashes = load_array(self.cache.path / "group_hashes.npy")

    def retrieve_smiles(self, smiles):
        from rdkit.Chem import rdFingerprintGenerator

        from structure_pretrain.library_builder.canonicalize import (
            nonisomeric_smiles,
            parse_molecule,
        )

        canonical = nonisomeric_smiles(parse_molecule(smiles))
        target_id = ("pl_" + hashlib.sha256(canonical.encode()).hexdigest()).encode()
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=False
        )
        binary = DataStructs.BitVectToBinaryText(
            generator.GetFingerprint(parse_molecule(canonical))
        )
        vector = np.frombuffer(binary, dtype=np.uint8)
        key = np.frombuffer(hashlib.blake2b(binary, digest_size=16).digest(), dtype="V16")[0]
        group = int(np.searchsorted(self.hashes, key))
        own = (
            group < len(self.hashes)
            and self.hashes[group] == key
            and np.array_equal(vector, self.cache.vectors[group])
        )
        if own and self.cache.offsets[group + 1] - self.cache.offsets[group] >= 11:
            keys = [group]
        else:
            match = self.index.search(
                vector, count=min(self.settings["candidate_groups"], len(self.hashes))
            )
            keys = ([group] if own else []) + list(match.keys)
        hits = rank_groups(
            vector, keys, self.cache.vectors, self.cache.members, self.cache.ids, -1, k=11
        )
        return [
            {
                "row_index": row,
                "structure_id": bytes(self.cache.ids[row]).decode(),
                "structure_similarity": score,
            }
            for row, score in hits
            if bytes(self.cache.ids[row]) != target_id
        ][:10]


def make_index(vectors, keys=None, expansion_search=128, threads=4):
    index = Index(
        ndim=2048,
        metric="tanimoto",
        dtype="b1",
        connectivity=32,
        expansion_add=256,
        expansion_search=expansion_search,
        enable_key_lookups=False,
    )
    keys = np.arange(len(vectors), dtype=np.uint64) if keys is None else keys
    for start in range(0, len(vectors), 8192):
        index.add(keys[start : start + 8192], vectors[start : start + 8192], threads=threads)
    return index


def rank_groups(query_fp, group_keys, vectors, members, ids, exclude_row, k=10, threshold=0.4):
    keys = list(dict.fromkeys(int(g) for g in group_keys))
    query = DataStructs.CreateFromBinaryText(bytes(query_fp))
    refs = [DataStructs.CreateFromBinaryText(bytes(vectors[g])) for g in keys]
    scores = DataStructs.BulkTanimotoSimilarity(query, refs)
    candidates = []
    for group, score in zip(keys, scores):
        if score < threshold:
            continue
        for row in members(group, k + 1):
            row = int(row)
            if row != exclude_row:
                candidates.append((row, float(score)))
    # One canonical graph is one library row; FP collisions remain distinct usable templates.
    return sorted(candidates, key=lambda hit: (-hit[1], bytes(ids[hit[0]])))[:k]


def exact_benchmark(cache, output, target_count=10000, reference_count=20000, seed=42):
    output = Path(output)
    n = len(cache.fp)
    if target_count > reference_count or reference_count > n:
        raise ValueError("Benchmark target/reference subset sizes invalid")
    rng = np.random.default_rng(seed)
    reference_rows = rng.choice(n, reference_count, replace=False)
    target_rows = reference_rows[:target_count]
    np.save(output / "benchmark_reference_rows.npy", reference_rows, allow_pickle=False)
    np.save(output / "benchmark_target_rows.npy", target_rows, allow_pickle=False)
    groups = np.unique(cache.group_of_row[reference_rows])
    postings = {}
    for row in reference_rows:
        postings.setdefault(int(cache.group_of_row[row]), []).append(int(row))
    postings = {
        g: sorted(rows, key=lambda row: bytes(cache.ids[row])) for g, rows in postings.items()
    }

    def members(group, count):
        return postings[group][:count]

    vectors = np.asarray(cache.vectors[groups])
    index = make_index(vectors, groups.astype(np.uint64))
    refs = [DataStructs.CreateFromBinaryText(bytes(cache.fp[row])) for row in reference_rows]
    ref_ids = np.asarray(cache.ids[reference_rows])
    gold, gold_scores = [], []
    start = time.monotonic()
    for i, row in enumerate(target_rows):
        scores = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                DataStructs.CreateFromBinaryText(bytes(cache.fp[row])), refs
            )
        )
        scores[reference_rows == row] = -1
        order = np.lexsort((ref_ids, -scores))
        order = order[scores[order] >= 0.4][:10]
        gold.append(reference_rows[order])
        gold_scores.append(
            (
                set(int(r) for r in reference_rows[scores > scores[order[-1]]]),
                float(scores[order[-1]]),
                float(scores[order[0]]),
                float(scores[order].sum()),
            )
            if len(order)
            else (set(), -1.0, -1.0, 0.0)
        )
        if (i + 1) % 1000 == 0:
            print(f"exact benchmark {i + 1}/{target_count}", flush=True)
    exact_seconds = time.monotonic() - start
    attempts = []
    for expansion, candidate_groups in ((128, 64), (256, 128), (512, 256), (1024, 512)):
        index.expansion_search = expansion
        candidate_groups = min(candidate_groups, len(groups))
        matches = index.search(np.asarray(cache.fp[target_rows]), count=candidate_groups, threads=4)
        strict, tie_aware, top1, regret = [], [], [], []
        for i, row in enumerate(target_rows):
            hits = rank_groups(
                cache.fp[row],
                [int(cache.group_of_row[row]), *matches.keys[i, : matches.counts[i]]],
                cache.vectors,
                members,
                cache.ids,
                int(row),
            )
            truth = gold[i]
            selected = {r for r, _ in hits}
            if not len(truth):
                strict.append(float(not hits))
                tie_aware.append(float(not hits))
                top1.append(float(not hits))
                regret.append(0.0)
                continue
            higher, boundary, best_score, gold_sum = gold_scores[i]
            boundary_hit_count = sum(abs(score - boundary) < 1e-12 for _, score in hits)
            credit = len(selected & higher) + min(boundary_hit_count, len(truth) - len(higher))
            strict.append(len(selected & set(truth.tolist())) / len(truth))
            tie_aware.append(credit / len(truth))
            top1.append(float(bool(hits) and hits[0][1] >= best_score - 1e-12))
            regret.append(max(0.0, (gold_sum - sum(s for _, s in hits)) / len(truth)))
        result = {
            "expansion_search": expansion,
            "candidate_groups": candidate_groups,
            "strict_id_recall_at_10": float(np.mean(strict)),
            "tie_aware_recall_at_10": float(np.mean(tie_aware)),
            "top1_score_recall": float(np.mean(top1)),
            "mean_score_regret_at_10": float(np.mean(regret)),
        }
        attempts.append(result)
        if result["tie_aware_recall_at_10"] >= 0.99 and result["top1_score_recall"] >= 0.995:
            break
    passed = (
        attempts[-1]["tie_aware_recall_at_10"] >= 0.99
        and attempts[-1]["top1_score_recall"] >= 0.995
    )
    report = {
        "status": "passed" if passed else "failed",
        "seed": seed,
        "target_count": target_count,
        "reference_subset_count": reference_count,
        "reference_unique_fingerprints": len(groups),
        "index_parameters": {"connectivity": 32, "expansion_add": 256},
        "exact_tanimoto_comparisons": target_count * reference_count,
        "exact_seconds": exact_seconds,
        "attempts": attempts,
        "selected_settings": attempts[-1],
        "tie_aware_definition": (
            "All scores above the exact kth score must be recalled; any identity tied "
            "at the kth score may fill remaining slots"
        ),
        "scope": "Random benchmark reference subset, not a proof of full-library exact recall",
        "training_or_validation_split_created": False,
        "identity_exclusion_verified": True,
    }
    dump_json(output / "exact_benchmark.json", report)
    if not passed:
        raise ValueError("ANN recall benchmark failed; full-library readiness audit blocked")
    return report


def distribution(values):
    values = np.asarray(values)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
        },
    }


def summarize_metrics(counts, maximum, mean):
    n = len(counts)
    usable = counts > 0
    return {
        "targets": n,
        "at_least_1": int(np.count_nonzero(usable)),
        "at_least_1_fraction": float(np.mean(usable)),
        "at_least_5_fraction": float(np.mean(counts >= 5)),
        "ten_fraction": float(np.mean(counts == 10)),
        "zero_template_count": int(np.count_nonzero(~usable)),
        "max_similarity": distribution(maximum[usable]),
        "top10_mean_similarity": distribution(mean[usable]),
        "mean_definition": "Mean over available templates up to 10; no zero padding",
    }


def full_readiness(cache, benchmark, output):
    output = Path(output)
    settings = benchmark["selected_settings"]
    index_path = output / "morgan_groups.usearch"
    if index_path.exists():
        index = Index.restore(index_path, view=True)
    else:
        start = time.monotonic()
        index = make_index(cache.vectors, expansion_search=settings["expansion_search"])
        index.save(index_path)
        print(
            f"full HNSW index: {len(cache.vectors)} groups; "
            f"built in {time.monotonic() - start:.1f}s",
            flush=True,
        )
    index.expansion_search = settings["expansion_search"]
    sizes = np.diff(cache.offsets)
    counts = np.zeros(len(sizes), dtype=np.uint8)
    maximum, means = np.zeros(len(sizes), dtype=np.float32), np.zeros(len(sizes), dtype=np.float32)
    large = sizes >= 11
    counts[large] = 10
    maximum[large] = 1
    means[large] = 1
    todo = np.flatnonzero(~large)
    candidate_count = min(settings["candidate_groups"], len(sizes))
    for start in range(0, len(todo), 512):
        groups = todo[start : start + 512]
        matches = index.search(np.asarray(cache.vectors[groups]), count=candidate_count, threads=4)
        for i, group in enumerate(groups):
            row = int(cache.members(int(group), 1)[0])
            hits = rank_groups(
                cache.fp[row],
                [int(group), *matches.keys[i, : matches.counts[i]]],
                cache.vectors,
                cache.members,
                cache.ids,
                row,
            )
            counts[group] = len(hits)
            if hits:
                maximum[group] = hits[0][1]
                means[group] = sum(s for _, s in hits) / len(hits)
        if start % 10240 == 0:
            dump_json(
                output / "progress.json",
                {
                    "stage": "retrieval_readiness",
                    "groups_done": start + len(groups),
                    "groups_requiring_ann": len(todo),
                    "exact_identical_fp_groups": int(large.sum()),
                },
            )
            print(
                f"readiness {start + len(groups)}/{len(todo)} nontrivial fingerprint groups",
                flush=True,
            )
    row_count, row_max, row_mean = (
        counts[cache.group_of_row],
        maximum[cache.group_of_row],
        means[cache.group_of_row],
    )
    np.savez_compressed(
        output / "target_readiness_metrics.npz",
        template_count=row_count,
        max_similarity=row_max,
        top10_mean_similarity=row_mean,
    )
    categories = {
        "chemistry_family": (0, FAMILIES),
        "plausibility_tier": (1, TIERS),
        "structure_origin": (2, ORIGINS),
    }
    grouped = {}
    for name, (column, names) in categories.items():
        grouped[name] = {
            label: summarize_metrics(row_count[mask], row_max[mask], row_mean[mask])
            for code, label in enumerate(names)
            if np.any(mask := cache.labels[:, column] == code)
        }
    source_masks = {
        "PubChem-derived": cache.labels[:, 2] <= 1,
        "generated_phosphate_analog": cache.labels[:, 2] == 2,
        "generated_keto_inositol": cache.labels[:, 2] == 3,
    }
    grouped["source_group"] = {
        name: summarize_metrics(row_count[mask], row_max[mask], row_mean[mask])
        for name, mask in source_masks.items()
        if mask.any()
    }
    report = {
        "status": "complete",
        "method": (
            "USearch Tanimoto HNSW over unique packed Morgan bit vectors, "
            "exact RDKit reranking, distinct canonical-graph postings"
        ),
        "fingerprint": {"radius": 2, "bits": 2048, "includeChirality": False},
        "max_top_k": 10,
        "minimum_similarity": 0.4,
        "identical_canonical_target_excluded": True,
        "identical_fp_shortcut_exact": True,
        "benchmark": settings,
        "overall": summarize_metrics(row_count, row_max, row_mean),
        "grouped": grouped,
        "all_pairs_full_library_computation_performed": False,
        "readiness_is_approximate_search_based": True,
        "training_performed": False,
        "dataset_split_created": False,
        "structures_sha256": cache.manifest["structures_sha256"],
    }
    dump_json(output / "retrieval_readiness_report.json", report)
    dump_json(output / "progress.json", {"stage": "complete", "targets": len(row_count)})
    return report
