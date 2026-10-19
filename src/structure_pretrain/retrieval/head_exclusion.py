"""Paired identity-only / entire-polar-id-excluded retrieval on a frozen library."""

import json
import multiprocessing
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import DataStructs
from usearch.index import Index

from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_plausibility.pipeline import dump_json

from .indexed_morgan import (
    FAMILIES,
    FingerprintCache,
    distribution,
    load_array,
    make_index,
    search_many,
    summarize_metrics,
)

THRESHOLD = 0.4
K = 10
_WORKER_CACHE = _WORKER_CODES = _WORKER_MAX_PER_HEAD = None


def prepare_heads(structures, cache, output):
    """Numeric exact polar IDs aligned to immutable cache row IDs, verified row-by-row."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    digest = file_hash(Path(structures))
    if digest != cache.manifest["structures_sha256"]:
        raise ValueError("Polar labels and fingerprint cache must use the exact same source")
    manifest_path = output / "polar_codes_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["structures_sha256"] != digest or manifest["status"] != "complete":
            raise ValueError("Head-label cache source changed")
        return load_array(output / "polar_codes.npy"), manifest
    codes = np.lib.format.open_memmap(
        output / "polar_codes.npy", mode="w+", dtype=np.uint32, shape=(len(cache.fp),)
    )
    vocabulary, counts = {}, Counter()
    offset = 0
    for batch in pq.ParquetFile(structures).iter_batches(
        batch_size=10000, columns=["structure_id", "polar_id"]
    ):
        rows = batch.to_pydict()
        ids = np.asarray(rows["structure_id"], dtype="S67")
        if not np.array_equal(ids, cache.ids[offset : offset + len(ids)]):
            raise ValueError("Fingerprint cache row order differs from library")
        for j, head in enumerate(rows["polar_id"]):
            code = vocabulary.setdefault(head, len(vocabulary))
            codes[offset + j] = code
            counts[code] += 1
        offset += len(ids)
    codes.flush()
    names = np.array(list(vocabulary), dtype=f"U{max(map(len, vocabulary))}")
    np.save(output / "polar_ids.npy", names, allow_pickle=False)
    manifest = {
        "status": "complete",
        "structures_sha256": digest,
        "row_count": offset,
        "unique_polar_ids": len(vocabulary),
        "max_structures_per_polar": max(counts.values()),
        "polar_codes_sha256": file_hash(output / "polar_codes.npy"),
        "all_row_ids_and_order_verified": True,
    }
    dump_json(manifest_path, manifest)
    return codes, manifest


def candidates_from_groups(keys, scores, cache, limit, members=None):
    """Postings are ID-sorted; scan beyond forbidden-head prefixes when necessary."""
    rows, values = [], []
    for group, score in zip(keys, scores):
        if score < THRESHOLD - 1e-7:
            continue
        posting = (
            cache.members(int(group), limit) if members is None else members[int(group)][:limit]
        )
        rows.extend(int(r) for r in posting)
        values.extend([float(score)] * len(posting))
    return np.asarray(rows, dtype=np.int64), np.asarray(values, dtype=np.float64)


def choose(rows, scores, ids, codes, forbidden_head=None, exclude_row=-1, k=K):
    mask = (rows != exclude_row) & (scores >= THRESHOLD - 1e-7)
    if forbidden_head is not None:
        mask &= codes[rows] != forbidden_head
    rows, scores = rows[mask], scores[mask]
    order = np.lexsort((ids[rows], -scores))[:k]
    return [(int(rows[i]), float(scores[i])) for i in order]


def exact_rerank(hits, query_group, cache):
    if not hits:
        return []
    groups = list(dict.fromkeys(int(cache.group_of_row[row]) for row, _ in hits))
    scores = DataStructs.BulkTanimotoSimilarity(
        cache.rdkit_group(int(query_group)), [cache.rdkit_group(g) for g in groups]
    )
    by_group = dict(zip(groups, scores))
    return sorted(
        [
            (row, float(by_group[int(cache.group_of_row[row])]))
            for row, _ in hits
            if by_group[int(cache.group_of_row[row])] >= THRESHOLD
        ],
        key=lambda hit: (-hit[1], bytes(cache.ids[hit[0]])),
    )


def paired_group(cache, codes, group, keys, distances, max_per_head, members=None, targets=None):
    """One ANN query serves all FP-identical targets; exclusions still depend on head ID.

    Native B1 Tanimoto equals RDKit bit Tanimoto up to floating point rounding.
    Use native scores to order candidate rows, then exact-rerank every selected
    group with RDKit. More than max_per_head+K group/posting slots ensures a
    forbidden head cannot consume the search prefix. Include boundary ties from
    the entire ANN candidate list for the head-excluded arm.
    """
    keys = np.asarray(keys, dtype=np.int64)
    scores = 1 - np.asarray(distances, dtype=np.float64)
    # Prepend own group exactly once, even if HNSW did not return it.
    keep = keys != group
    keys, scores = np.r_[group, keys[keep]], np.r_[1.0, scores[keep]]
    targets = (
        cache.members(group, int(cache.offsets[group + 1] - cache.offsets[group]))
        if targets is None
        else targets
    )
    # Reproduce the frozen rank_groups identity-only 16-group candidate policy.
    base_rows, base_scores = candidates_from_groups(keys[:16], scores[:16], cache, 11, members)
    baseline = choose(base_rows, base_scores, cache.ids, codes, k=11)
    baseline = exact_rerank(baseline, group, cache)
    # Each forbidden head occupies <= max_per_head canonical rows globally.
    # A prefix of max_per_head+K+1 distinct groups is sufficient for counts.
    depth = min(len(keys), max(32, max_per_head + K + 1))
    cut = scores[depth - 1] if depth else 1.0
    depth = int(np.count_nonzero(scores >= cut - 1e-7))
    selected_keys, selected_scores = keys[:depth], scores[:depth]
    rows, values = candidates_from_groups(
        selected_keys, selected_scores, cache, max_per_head + K, members
    )
    excluded, base_by_target = {}, {}
    for head in np.unique(codes[targets]):
        hits = choose(rows, values, cache.ids, codes, forbidden_head=int(head))
        excluded[int(head)] = exact_rerank(hits, group, cache)
    for row in targets:
        base_by_target[int(row)] = [(r, s) for r, s in baseline if r != row][:K]
    return base_by_target, excluded


def exact_excluded(cache, codes, group, forbidden_head, max_per_head, chunk_size=8192):
    """Confirm every rare <10 result against all vectors, without N x N work."""
    query = cache.rdkit_group(int(group))
    best = []
    for start in range(0, len(cache.vectors), chunk_size):
        groups = np.arange(start, min(start + chunk_size, len(cache.vectors)), dtype=np.int64)
        refs = [DataStructs.CreateFromBinaryText(bytes(cache.vectors[int(g)])) for g in groups]
        scores = np.asarray(DataStructs.BulkTanimotoSimilarity(query, refs))
        keep = np.flatnonzero(scores >= THRESHOLD)
        # Stream top K canonical rows after head exclusion; never prune a group
        # merely because a previous, forbidden group scored higher.
        rows, values = candidates_from_groups(groups[keep], scores[keep], cache, max_per_head + K)
        hits = choose(rows, values, cache.ids, codes, forbidden_head=forbidden_head)
        best.extend(hits)
        best = sorted(best, key=lambda hit: (-hit[1], bytes(cache.ids[hit[0]])))[:K]
    return best


def exclusion_benchmark(
    cache, codes, manifest, output, target_count=2000, reference_count=20000, threads=2, seed=107
):
    """Head-enriched exact benchmark; random independent rows hide head leakage."""
    output = Path(output)
    n = len(cache.fp)
    rng = np.random.default_rng(seed)
    reference_count, target_count = min(reference_count, n), min(target_count, n)
    seeds = rng.choice(n, min(target_count // 2, n), replace=False)
    selected_heads = np.unique(codes[seeds])
    same_head_rows = np.flatnonzero(np.isin(codes, selected_heads))
    if len(same_head_rows) > reference_count:
        same_head_rows = same_head_rows[:reference_count]
    extra = rng.choice(n, reference_count, replace=False)
    refs = np.unique(np.r_[same_head_rows, extra])
    if len(refs) > reference_count:
        refs = np.r_[
            same_head_rows,
            np.setdiff1d(extra, same_head_rows)[: reference_count - len(same_head_rows)],
        ]
    targets = np.r_[np.intersect1d(seeds, refs), rng.permutation(np.setdiff1d(refs, seeds))][
        :target_count
    ]
    np.savez(output / "exclusion_benchmark_rows.npz", references=refs, targets=targets)
    postings = {}
    for row in refs:
        postings.setdefault(int(cache.group_of_row[row]), []).append(int(row))
    postings = {
        g: sorted(rows, key=lambda row: bytes(cache.ids[row])) for g, rows in postings.items()
    }
    groups = np.array(sorted(postings), dtype=np.uint64)
    index = make_index(
        np.asarray(cache.vectors[groups]), groups, expansion_search=1024, threads=threads
    )
    ref_fps = [DataStructs.CreateFromBinaryText(bytes(cache.fp[r])) for r in refs]
    matches = search_many(
        index, np.asarray(cache.fp[targets]), count=min(512, len(groups)), threads=threads
    )
    strict, tie, top1, regret, native_error = [], [], [], [], []
    started = time.monotonic()
    for i, row in enumerate(targets):
        scores = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                cache.rdkit_group(int(cache.group_of_row[row])), ref_fps
            )
        )
        scores[codes[refs] == codes[row]] = -1
        order = np.lexsort((cache.ids[refs], -scores))
        truth = order[scores[order] >= THRESHOLD][:K]
        _, by_head = paired_group(
            cache,
            codes,
            int(cache.group_of_row[row]),
            matches.keys[i, : matches.counts[i]],
            matches.distances[i, : matches.counts[i]],
            manifest["max_structures_per_polar"],
            postings,
            np.array([row]),
        )
        hits = by_head[int(codes[row])]
        chosen = {r for r, _ in hits}
        if not len(truth):
            strict.append(float(not hits))
            tie.append(float(not hits))
            top1.append(float(not hits))
            regret.append(0.0)
        else:
            boundary, best = scores[truth[-1]], scores[truth[0]]
            higher = set(int(refs[j]) for j in np.flatnonzero(scores > boundary))
            credit = len(chosen & higher) + min(
                sum(abs(s - boundary) < 1e-12 for _, s in hits), len(truth) - len(higher)
            )
            strict.append(len(chosen & set(refs[truth])) / len(truth))
            tie.append(credit / len(truth))
            top1.append(float(bool(hits) and hits[0][1] >= best - 1e-12))
            regret.append(max(0.0, (scores[truth].sum() - sum(s for _, s in hits)) / len(truth)))
        if i < 100:
            keys = matches.keys[i, : matches.counts[i]]
            exact = DataStructs.BulkTanimotoSimilarity(
                cache.rdkit_group(int(cache.group_of_row[row])),
                [cache.rdkit_group(int(g)) for g in keys],
            )
            native_error.append(
                float(
                    np.max(
                        np.abs(np.asarray(exact) - (1 - matches.distances[i, : matches.counts[i]]))
                    )
                )
            )
        if (i + 1) % 200 == 0:
            print(f"head-exclusion exact benchmark {i + 1}/{len(targets)}", flush=True)
    result = {
        "status": "passed"
        if np.mean(tie) >= 0.99 and np.mean(top1) >= 0.995 and max(native_error, default=0) < 1e-6
        else "failed",
        "target_count": len(targets),
        "reference_count": len(refs),
        "seed": seed,
        "same_head_enriched_reference_rows": len(same_head_rows),
        "exact_comparisons": len(targets) * len(refs),
        "seconds": time.monotonic() - started,
        "strict_id_recall_at_10": float(np.mean(strict)),
        "tie_aware_recall_at_10": float(np.mean(tie)),
        "top1_score_recall": float(np.mean(top1)),
        "mean_score_regret_at_10": float(np.mean(regret)),
        "native_vs_rdkit_max_absolute_error": max(native_error, default=0),
        "settings": {
            "expansion_search": 1024,
            "candidate_groups": 512,
            "adaptive_search_ceiling": 4096,
            "postings_are_head_filtered_before_truncation": True,
        },
        "scope": (
            "Head-enriched benchmark subset; full-library search remains ANN, selected scores exact"
        ),
        "training_or_validation_split_created": False,
    }
    dump_json(output / "head_exclusion_benchmark.json", result)
    if result["status"] != "passed":
        raise ValueError("Exclusion-aware exact benchmark failed")
    return result


def metric_arm(hits, scores):
    counts = np.count_nonzero(hits >= 0, axis=1).astype(np.uint8)
    maximum = np.max(np.where(hits >= 0, scores, 0), axis=1)
    means = np.divide(
        np.where(hits >= 0, scores, 0).sum(axis=1),
        counts,
        out=np.zeros(len(counts), dtype=np.float32),
        where=counts > 0,
    )
    return counts, maximum, means


def same_head_summary(hits, scores, codes, mask=None):
    target_rows = np.arange(len(codes)) if mask is None else np.flatnonzero(mask)
    same = np.zeros(len(target_rows), dtype=np.uint8)
    counts = np.zeros(len(target_rows), dtype=np.uint8)
    top1 = np.zeros(len(target_rows), dtype=bool)
    for start in range(0, len(target_rows), 10000):
        rows = target_rows[start : start + 10000]
        block = hits[rows]
        valid = block >= 0
        matches = valid & (codes[np.maximum(block, 0)] == codes[rows, None])
        same[start : start + len(rows)] = matches.sum(axis=1)
        counts[start : start + len(rows)] = valid.sum(axis=1)
        top1[start : start + len(rows)] = matches[:, 0]
    usable = counts > 0
    return {
        "targets": len(target_rows),
        "retrieved_template_pairs": int(counts.sum()),
        "same_polar_id_pairs": int(same.sum()),
        "same_polar_id_pair_fraction": float(same.sum() / counts.sum()) if counts.sum() else 0.0,
        "targets_with_any_same_polar_id_fraction": float(np.mean(same > 0)),
        "top1_same_polar_id_fraction_among_usable": float(top1[usable].mean())
        if usable.any()
        else 0.0,
        "per_target_same_polar_fraction": distribution(same[usable] / counts[usable]),
    }


def init_rank_worker(cache_path, codes_path, max_per_head):
    global _WORKER_CACHE, _WORKER_CODES, _WORKER_MAX_PER_HEAD
    _WORKER_CACHE = FingerprintCache(cache_path)
    _WORKER_CODES = load_array(codes_path)
    _WORKER_MAX_PER_HEAD = max_per_head


def rank_batch(task, cache=None, codes=None, max_per_head=None):
    cache = _WORKER_CACHE if cache is None else cache
    codes = _WORKER_CODES if codes is None else codes
    max_per_head = _WORKER_MAX_PER_HEAD if max_per_head is None else max_per_head
    groups, keys, distances, counts = task
    results = []
    for i, group in enumerate(groups):
        baseline, excluded = paired_group(
            cache, codes, int(group), keys[i, : counts[i]], distances[i, : counts[i]], max_per_head
        )
        results.append((int(group), baseline, excluded))
    return int(groups[-1]) + 1, results


def group_batches(cache, codes, index, start_group, max_per_head, threads, workers, codes_path):
    def tasks():
        for start in range(start_group, len(cache.vectors), 256):
            groups = np.arange(start, min(start + 256, len(cache.vectors)), dtype=np.int64)
            matches = search_many(
                index,
                np.asarray(cache.vectors[groups]),
                count=min(512, len(cache.vectors)),
                threads=threads,
            )
            yield groups, matches.keys, matches.distances, matches.counts

    if not workers:
        for task in tasks():
            yield rank_batch(task, cache, codes, max_per_head)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_rank_worker,
        initargs=(cache.path, codes_path, max_per_head),
    ) as pool:
        pending = deque()
        for task in tasks():
            pending.append(pool.submit(rank_batch, task))
            if len(pending) >= workers * 2:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def full_head_audit(
    structures, cache_path, output, threads=2, benchmark_targets=2000, workers=0, working_cache=None
):
    output = Path(output)
    cache = FingerprintCache(working_cache or cache_path)
    codes, manifest = prepare_heads(structures, cache, output)
    benchmark_file = output / "head_exclusion_benchmark.json"
    benchmark = (
        json.loads(benchmark_file.read_text(encoding="utf-8"))
        if benchmark_file.exists()
        else exclusion_benchmark(cache, codes, manifest, output, benchmark_targets, threads=threads)
    )
    if benchmark["status"] != "passed":
        raise ValueError("Passing exclusion benchmark required")
    index_path = cache.path / "morgan_groups.usearch"
    index = Index.restore(index_path, view=True)
    index.expansion_search = 1024
    signature = {
        "structures_sha256": cache.manifest["structures_sha256"],
        "index_sha256": file_hash(index_path),
        "polar_codes_sha256": manifest["polar_codes_sha256"],
        "threshold": THRESHOLD,
        "k": K,
        "settings": benchmark["settings"],
    }
    checkpoint = output / "checkpoint.json"
    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else None
    if state and state["signature"] != signature:
        raise ValueError("Resume inputs/settings changed")
    arrays = {}
    for name in ("baseline_rows", "excluded_rows", "baseline_scores", "excluded_scores"):
        dtype = np.int32 if name.endswith("rows") else np.float32
        arrays[name] = np.lib.format.open_memmap(
            output / f"{name}.npy",
            mode="r+" if state else "w+",
            dtype=dtype,
            shape=(len(cache.fp), K),
        )
        if not state:
            arrays[name][:] = -1 if name.endswith("rows") else np.nan
    start_group = state["next_group"] if state else 0
    corrections = state.get("exact_low_count_confirmations", []) if state else []
    max_per_head = manifest["max_structures_per_polar"]
    started = time.monotonic()
    last_log = started
    for end_group, batch_results in group_batches(
        cache, codes, index, start_group, max_per_head, threads, workers, output / "polar_codes.npy"
    ):
        for group, baseline, excluded in batch_results:
            targets = cache.members(
                int(group), int(cache.offsets[group + 1] - cache.offsets[group])
            )
            rare = [head for head, hits in excluded.items() if len(hits) < K]
            if rare:
                for depth in (1024, 2048, 4096):
                    match = index.search(
                        np.asarray(cache.vectors[group]), count=min(depth, len(cache.vectors))
                    )
                    _, larger = paired_group(
                        cache, codes, int(group), match.keys, match.distances, max_per_head
                    )
                    for head in rare:
                        excluded[head] = larger[head]
                    rare = [head for head in rare if len(excluded[head]) < K]
                    if not rare or depth >= len(cache.vectors):
                        break
                for head in rare:
                    previous = len(excluded[head])
                    excluded[head] = exact_excluded(cache, codes, group, head, max_per_head)
                    corrections.append(
                        {
                            "group": int(group),
                            "polar_code": int(head),
                            "ann_count": previous,
                            "exact_count": len(excluded[head]),
                        }
                    )
                    print(
                        f"exact low-count confirmation group={group}, head={head}: "
                        f"{previous}->{len(excluded[head])}",
                        flush=True,
                    )
            for row in targets:
                for arm, hits in (
                    ("baseline", baseline[int(row)]),
                    ("excluded", excluded[int(codes[row])]),
                ):
                    arrays[f"{arm}_rows"][row] = -1
                    arrays[f"{arm}_scores"][row] = np.nan
                    if hits:
                        arrays[f"{arm}_rows"][row, : len(hits)] = [r for r, _ in hits]
                        arrays[f"{arm}_scores"][row, : len(hits)] = [s for _, s in hits]
        now = time.monotonic()
        if now - last_log >= 30 or end_group == len(cache.vectors):
            for array in arrays.values():
                array.flush()
            dump_json(
                checkpoint,
                {
                    "signature": signature,
                    "next_group": end_group,
                    "total_groups": len(cache.vectors),
                    "exact_low_count_confirmations": corrections,
                },
            )
            print(
                f"paired head audit {end_group}/{len(cache.vectors)} "
                f"fingerprint groups; elapsed {now - started:.1f}s",
                flush=True,
            )
            last_log = now
    results = {}
    for arm in ("baseline", "excluded"):
        hits, scores = arrays[f"{arm}_rows"], arrays[f"{arm}_scores"]
        counts, maximum, means = metric_arm(hits, scores)
        np.savez_compressed(
            output / f"{arm}_target_metrics.npz",
            template_count=counts,
            max_similarity=maximum,
            top10_mean_similarity=means,
        )
        results[arm] = {
            "overall": summarize_metrics(counts, maximum, means),
            "same_head": same_head_summary(hits, scores, codes),
            "by_chemistry_family": {
                label: summarize_metrics(counts[mask], maximum[mask], means[mask])
                for i, label in enumerate(FAMILIES)
                if np.any(mask := cache.labels[:, 0] == i)
            },
        }
    # Full-row invariants include the exact label exclusion, distinct templates,
    # no self matches, and finite thresholded scores for every stored hit.
    checked = 0
    for start in range(0, len(codes), 10000):
        target = np.arange(start, min(start + 10000, len(codes)))
        for arm in ("baseline", "excluded"):
            rows, scores = arrays[f"{arm}_rows"][target], arrays[f"{arm}_scores"][target]
            valid = rows >= 0
            assert not np.any(valid & (rows == target[:, None]))
            assert np.all(np.isfinite(scores[valid]) & (scores[valid] >= THRESHOLD - 1e-7))
            for column in range(K - 1):
                assert not np.any(
                    (rows[:, column, None] == rows[:, column + 1 :]) & (rows[:, column, None] >= 0)
                )
            if arm == "excluded":
                assert not np.any(valid & (codes[np.maximum(rows, 0)] == codes[target, None]))
        checked += len(target)
    original = json.loads(
        (Path(cache_path) / "retrieval_readiness_report.json").read_text(encoding="utf-8")
    )
    old = np.load(Path(cache_path) / "target_readiness_metrics.npz", allow_pickle=False)
    new = np.load(output / "baseline_target_metrics.npz", allow_pickle=False)
    reproduction = {
        key: {
            "equal_rows": int(np.count_nonzero(np.isclose(new[key], old[key], atol=1e-6, rtol=0))),
            "max_absolute_difference": float(
                np.max(np.abs(new[key].astype(float) - old[key].astype(float)))
            ),
        }
        for key in new.files
    }
    report = {
        "status": "complete",
        "target_reference_scope": "Frozen v0.2, all original rows as targets and references",
        **signature,
        "fingerprint": {"radius": 2, "bits": 2048, "includeChirality": False},
        "benchmark": benchmark,
        "baseline_identity_only": results["baseline"],
        "exclude_all_same_polar_id": results["excluded"],
        "original_full_retrieval_overall": original["overall"],
        "baseline_metric_reproduction": reproduction,
        "all_target_rows_exclusion_and_identity_checked": checked,
        "exact_low_count_confirmations": corrections,
        "scope": (
            "ANN candidates, RDKit exact selected similarities; "
            "all <10 head-excluded results confirmed exhaustively"
        ),
        "full_neighbor_rows_and_scores_saved": True,
        "row_index_resolves_via": str((Path(cache_path) / "row_ids.npy").resolve()),
        "training_performed": False,
        "dataset_split_created": False,
    }
    dump_json(output / "head_exclusion_report.json", report)
    return report
