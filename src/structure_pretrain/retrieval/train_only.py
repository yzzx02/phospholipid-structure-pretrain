"""Separate train-only reference index and auditable normal/cross-polar caches."""

import argparse
import time
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
from rdkit import DataStructs
from usearch.index import Index

from structure_pretrain.data.prepare_v03 import SCHEMA, SPLITS, read_json
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_plausibility.pipeline import dump_json
from structure_pretrain.retrieval.head_exclusion import paired_group
from structure_pretrain.retrieval.indexed_morgan import (
    FingerprintCache,
    load_array,
    make_index,
    search_many,
    summarize_metrics,
)


class TrainCache(FingerprintCache):
    def __init__(self, frozen, prepared, bank):
        super().__init__(frozen)
        self.all_offsets, self.all_postings = self.offsets, self.postings
        self.offsets = load_array(Path(bank) / "posting_offsets.npy")
        self.postings = load_array(Path(bank) / "posting_rows.npy")
        self.train_groups = load_array(Path(bank) / "train_fp_groups.npy")
        self.heads = load_array(Path(prepared) / "polar_groups.npy")
        self.splits = load_array(Path(prepared) / "row_splits.npy")
        self.max_head = int(np.bincount(np.asarray(self.heads)).max())

    def targets(self, group):
        a, b = self.all_offsets[group : group + 2]
        return self.all_postings[a:b]

    @cached_property
    def bitcounts(self):
        counts = np.empty(len(self.vectors), dtype=np.uint16)
        for start in range(0, len(self.vectors), 8192):
            counts[start : start + 8192] = popcounts(self.vectors[start : start + 8192]).sum(1)
        return counts

    @lru_cache(maxsize=256)
    def exact_shortlist(self, query_group):
        """Scan ALL train bit vectors, retaining all ties at a proven group bound.

        At most max_head graph rows are forbidden by one polar graph, so
        max_head+K+1 nonempty train groups suffice for K legal graph rows.
        This confirmation does not use an ANN candidate shortlist.
        """
        query = self.vectors[query_group]
        depth = min(len(self.train_groups), self.max_head + 11)
        best_groups, best_scores = np.empty(0, np.int64), np.empty(0, np.float64)
        for start in range(0, len(self.train_groups), 8192):
            groups = self.train_groups[start : start + 8192]
            intersect = popcounts(np.bitwise_and(self.vectors[groups], query)).sum(1)
            union = (
                self.bitcounts[groups].astype(np.int32)
                + int(self.bitcounts[query_group])
                - intersect
            )
            scores = np.divide(intersect, union, out=np.ones(len(groups)), where=union != 0)
            cutoff = (
                max(0.4, float(np.partition(scores, len(scores) - depth)[-depth]))
                if len(scores) > depth
                else 0.4
            )
            keep = scores >= cutoff
            best_groups, best_scores = (
                np.r_[best_groups, groups[keep]],
                np.r_[best_scores, scores[keep]],
            )
            if len(best_scores) > depth:
                cutoff = np.partition(best_scores, len(best_scores) - depth)[-depth]
                keep = best_scores >= cutoff
                best_groups, best_scores = best_groups[keep], best_scores[keep]
        if len(best_groups):
            checked = DataStructs.BulkTanimotoSimilarity(
                self.rdkit_group(query_group), [self.rdkit_group(int(g)) for g in best_groups]
            )
            if not np.allclose(best_scores, checked, rtol=0, atol=1e-12):
                raise ValueError("Packed exact Tanimoto differs from RDKit")
            return tuple(zip(best_groups.tolist(), checked))
        return ()


POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def popcounts(array):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(array)
    return POPCOUNT[array]  # Also supports NumPy 1.26 ROCm environments.


def exact_query(cache, row, cross):
    """Exhaustive streaming train-only confirmation. Stable graph-ID tie breaking."""
    best = []
    head = cache.heads[row]
    for g, score in cache.exact_shortlist(int(cache.group_of_row[row])):
        a, b = cache.offsets[g : g + 2]
        rows = cache.postings[a:b]
        rows = rows[(rows != row) & ((cache.heads[rows] != head) if cross else True)][:10]
        best.extend((int(r), float(score)) for r in rows)
    return sorted(best, key=lambda h: (-h[1], bytes(cache.ids[h[0]])))[:10]


def build_bank(prepared, frozen, work, threads=2):
    prepared, frozen, work = map(Path, (prepared, frozen, work))
    labels = read_json(prepared / "labels_manifest.json")
    if (
        labels["status"] != "complete"
        or read_json(prepared / "reconstruction_audit.json")["status"] != "passed"
    ):
        raise ValueError("Labels/tokenization/reconstruction gate failed")
    work.mkdir(parents=True, exist_ok=True)
    source = FingerprintCache(frozen)
    if source.manifest["structures_sha256"] != labels["structures_sha256"]:
        raise ValueError("Fingerprint cache does not match v0.3 label row order")
    binding = {
        "schema": SCHEMA,
        "version": "v0.3",
        "structures_sha256": labels["structures_sha256"],
        "labels_manifest_sha256": file_hash(prepared / "labels_manifest.json"),
        "row_splits_sha256": file_hash(prepared / "row_splits.npy"),
        "canonical_polar_groups_sha256": file_hash(prepared / "polar_groups.npy"),
    }
    manifest = work / "train_only_bank_manifest.json"
    if manifest.exists():
        old = read_json(manifest)
        if any(old.get(k) != v for k, v in binding.items()):
            raise ValueError("Existing train-only bank binding differs")
        return old
    splits = load_array(prepared / "row_splits.npy")
    training = np.flatnonzero(splits == 0)
    groups = np.unique(source.group_of_row[training])
    # Filter postings BEFORE index construction; no held-out vector-only groups in HNSW.
    postings = source.postings[splits[source.postings] == 0]
    counts = np.bincount(source.group_of_row[training], minlength=len(source.vectors))
    np.save(work / "posting_rows.npy", postings)
    np.save(work / "posting_offsets.npy", np.r_[0, np.cumsum(counts)])
    np.save(work / "train_fp_groups.npy", groups)
    index_file = work / "train_groups.usearch"
    if not index_file.exists():
        started = time.monotonic()
        print(
            f"building TRAIN ONLY index: {len(training)} graphs / {len(groups)} FP groups",
            flush=True,
        )
        index = make_index(
            source.vectors[groups],
            keys=groups.astype(np.uint64),
            expansion_search=1024,
            threads=threads,
        )
        index.save(index_file.with_suffix(".partial"))
        index_file.with_suffix(".partial").replace(index_file)
        del index
        print(f"train-only index seconds={time.monotonic() - started:.1f}", flush=True)
    report = {
        **binding,
        "status": "complete",
        "train_structures": len(training),
        "train_fp_groups": len(groups),
        "held_out_reference_rows": 0,
        "radius": 2,
        "fpSize": 2048,
        "includeChirality": False,
        "threshold": 0.4,
        "k": 10,
        "expansion_search": 1024,
        "reference_scope": "train structures only, before search",
        "index_sha256": file_hash(index_file),
        "derived_sha256": {
            n: file_hash(work / n)
            for n in ("posting_rows.npy", "posting_offsets.npy", "train_fp_groups.npy")
        },
    }
    dump_json(manifest, report)
    dump_json(prepared / "train_only_bank_manifest.json", report)
    return report


def open_results(output, n, resume=False):
    result = {}
    for mode in ("normal", "polar_excluded"):
        folder = Path(output) / f"retrieval_{mode}"
        folder.mkdir(parents=True, exist_ok=True)
        result[mode] = (
            np.lib.format.open_memmap(
                folder / "template_rows.npy",
                mode="r+" if resume else "w+",
                dtype=np.int32,
                shape=(n, 10),
            ),
            np.lib.format.open_memmap(
                folder / "scores.npy",
                mode="r+" if resume else "w+",
                dtype=np.float32,
                shape=(n, 10),
            ),
        )
        if not resume:
            result[mode][0][:] = -1
            result[mode][1][:] = np.nan
    return result


def store_hits(result, mode, row, hits):
    indices, scores = result[mode]
    indices[row] = -1
    scores[row] = np.nan
    if hits:
        indices[row, : len(hits)] = [r for r, _ in hits]
        scores[row, : len(hits)] = [s for _, s in hits]


def tie_aware_recall(hits, truth):
    if not truth:
        return float(not hits)
    boundary = truth[-1][1]
    higher = {r for r, score in truth if score > boundary + 1e-12}
    recalled = len({r for r, _ in hits} & higher)
    tied = sum(abs(score - boundary) < 1e-12 for _, score in hits)
    return (recalled + min(tied, len(truth) - len(higher))) / len(truth)


def calibrate_search(cache, index, work, bank, threads):
    """Choose search effort using TRAIN-only exact queries; Morgan semantics stay fixed.

    A small full-bank calibration is an engineering recall check, not a population
    proof. All selected pairs are TRAIN; test is not used to tune search effort.
    """
    path = Path(work) / "train_search_calibration.json"
    previous_attempts = []
    if path.exists():
        report = read_json(path)
        if report["index_sha256"] != bank["index_sha256"]:
            raise ValueError("Train search calibration index/status mismatch")
        if (
            report["status"] == "passed"
            and report.get("quality_metric") == "tie_aware_recall_at_10"
        ):
            return report
        previous_attempts = report["attempts"]
    gold_path = Path(work) / "calibration_gold_private.json"
    if gold_path.exists():
        gold = read_json(gold_path)
        if gold["index_sha256"] != bank["index_sha256"]:
            raise ValueError("Calibration exact gold belongs to another train bank")
        targets, answers = gold["targets"], gold["answers"]
    else:
        training = np.flatnonzero(cache.splits == 0)
        rng = np.random.default_rng(42)
        targets = rng.choice(training, min(32, len(training)), replace=False).tolist()
        answers = []
        for i, row in enumerate(targets):
            answers.append(
                {
                    "normal": exact_query(cache, row, False),
                    "polar_excluded": exact_query(cache, row, True),
                }
            )
            if (i + 1) % 4 == 0:
                print(f"train-only exact calibration {i + 1}/{len(targets)}", flush=True)
        dump_json(
            gold_path,
            {
                "schema": SCHEMA,
                "version": "v0.3",
                "structures_sha256": bank["structures_sha256"],
                "index_sha256": bank["index_sha256"],
                "targets": targets,
                "answers": answers,
            },
        )
    max_head = int(np.bincount(np.asarray(cache.heads)).max())
    attempts = previous_attempts.copy()
    for effort, wanted in (
        (128, 64),
        (256, 128),
        (512, 256),
        (1024, 512),
        (2048, 1024),
        (4096, 2048),
        (8192, 4096),
    ):
        index.expansion_search = effort
        depth = min(len(cache.train_groups), max(wanted, max_head + 11))
        if any(
            a["expansion_search"] == effort
            and a["candidate_groups"] == depth
            and "tie_aware_recall_at_10" in a
            for a in attempts
        ):
            continue
        start = time.monotonic()
        matches = search_many(index, np.asarray(cache.fp[targets]), depth, threads=threads)
        top1, recalls, regrets, tie_recalls = [], [], [], []
        for i, row in enumerate(targets):
            group = int(cache.group_of_row[row])
            normal, cross = paired_group(
                cache,
                cache.heads,
                group,
                matches.keys[i, : matches.counts[i]],
                matches.distances[i, : matches.counts[i]],
                max_head,
                targets=np.array([row]),
            )
            for mode, hits in (
                ("normal", normal[row]),
                ("polar_excluded", cross[int(cache.heads[row])]),
            ):
                truth = answers[i][mode]
                tie_recalls.append(tie_aware_recall(hits, truth))
                if not truth:
                    top1.append(float(not hits))
                    recalls.append(float(not hits))
                    regrets.append(0.0)
                    continue
                scores = [s for _, s in hits]
                desired = [s for _, s in truth]
                top1.append(float(bool(scores) and abs(scores[0] - desired[0]) <= 1e-7))
                credit = sum(abs(a - b) <= 1e-7 for a, b in zip(scores, desired))
                recalls.append(credit / len(desired))
                regrets.append(max(0.0, (sum(desired) - sum(scores)) / len(desired)))
        attempt = {
            "expansion_search": effort,
            "candidate_groups": depth,
            "top1_score_recall": float(np.mean(top1)),
            "top10_rank_score_recall": float(np.mean(recalls)),
            "tie_aware_recall_at_10": float(np.mean(tie_recalls)),
            "mean_score_regret": float(np.mean(regrets)),
            "query_seconds": time.monotonic() - start,
        }
        attempts.append(attempt)
        print(f"train search calibration: {attempt}", flush=True)
        if attempt["top1_score_recall"] >= 0.995 and attempt["tie_aware_recall_at_10"] >= 0.99:
            break
    passed = (
        attempts[-1]["top1_score_recall"] >= 0.995
        and attempts[-1]["tie_aware_recall_at_10"] >= 0.99
    )
    report = {
        "status": "passed" if passed else "failed",
        "index_sha256": bank["index_sha256"],
        "schema": SCHEMA,
        "version": "v0.3",
        "structures_sha256": bank["structures_sha256"],
        "train_query_count": len(targets),
        "seed": 42,
        "test_queries": 0,
        "reference_count": len(cache.postings),
        "attempts": attempts,
        "selected": attempts[-1],
        "quality_metric": "tie_aware_recall_at_10",
        "scope": "small exact full-train-bank calibration, not exhaustive recall proof",
    }
    dump_json(path, report)
    if not passed:
        raise ValueError("Train-only ANN calibration failed")
    return report


def retrieve(prepared, frozen, work, threads=2, batch_size=256):
    prepared, frozen, work = map(Path, (prepared, frozen, work))
    bank = build_bank(prepared, frozen, work, threads)
    report_path = prepared / "retrieval_coverage_by_split.json"
    if report_path.exists() and read_json(report_path).get("status") == "complete":
        return read_json(report_path)
    cache = TrainCache(frozen, prepared, work)
    index = Index.restore(work / "train_groups.usearch", view=True)
    calibration = calibrate_search(cache, index, work, bank, threads)
    index.expansion_search = calibration["selected"]["expansion_search"]
    max_head = int(load_array(prepared / "group_row_counts.npy").max())
    if max_head != int(np.bincount(np.asarray(cache.heads)).max()):
        raise ValueError("Derived head row counts do not match audited row-to-head mapping")
    depth = calibration["selected"]["candidate_groups"]
    state_path = prepared / "retrieval_progress.json"
    start_group = 0
    state = None
    if state_path.exists():
        state = read_json(state_path)
        if state["bank_sha256"] != bank["index_sha256"]:
            raise ValueError("Resume retrieval bank mismatch")
        start_group = state["next_group"]
        if start_group and (
            state.get("expansion_search") != index.expansion_search
            or state.get("candidate_groups") != depth
        ):
            raise ValueError("Retrieval resume search settings changed; mixed caches forbidden")
    results = open_results(prepared, len(cache.fp), resume=state_path.exists())
    started = time.monotonic()
    exact_confirmations = state.get("exact_low_count_confirmations", []) if state else []
    for start in range(start_group, len(cache.vectors), batch_size):
        end = min(start + batch_size, len(cache.vectors))
        matches = search_many(index, np.asarray(cache.vectors[start:end]), depth, threads=threads)
        for j, group in enumerate(range(start, end)):
            targets = cache.targets(group)
            normal, excluded = paired_group(
                cache,
                cache.heads,
                group,
                matches.keys[j, : matches.counts[j]],
                matches.distances[j, : matches.counts[j]],
                max_head,
                targets=targets,
            )
            # paired_group uses TRAIN postings only; query target postings remain full-library.
            for row in targets:
                row = int(row)
                for mode, hits in (
                    ("normal", normal[row]),
                    ("polar_excluded", excluded[int(cache.heads[row])]),
                ):
                    if len(hits) < 10:
                        hits = exact_query(cache, row, mode == "polar_excluded")
                        exact_confirmations.append({"row": row, "mode": mode, "count": len(hits)})
                    if any(cache.splits[r] != 0 or r == row for r, _ in hits):
                        raise ValueError("Illegal reference escaped train bank")
                    if mode == "polar_excluded" and any(
                        cache.heads[r] == cache.heads[row] for r, _ in hits
                    ):
                        raise ValueError("Canonical polar alias leaked into cross-head retrieval")
                    store_hits(results, mode, row, hits)
        if start % (batch_size * 16) == 0 or end == len(cache.vectors):
            for arrays in results.values():
                for array in arrays:
                    array.flush()
            dump_json(
                state_path,
                {
                    "next_group": end,
                    "bank_sha256": bank["index_sha256"],
                    "total_groups": len(cache.vectors),
                    "expansion_search": index.expansion_search,
                    "candidate_groups": depth,
                    "exact_low_count_confirmations": exact_confirmations,
                },
            )
            print(
                f"legal retrieval {end}/{len(cache.vectors)} FP groups; "
                f"{time.monotonic() - started:.1f}s this run",
                flush=True,
            )
    # Full-bank exact score benchmark of a fixed metadata-only target sample.
    rng = np.random.default_rng(42)
    benchmark = []
    for row in rng.choice(len(cache.fp), min(12, len(cache.fp)), replace=False):
        for mode in results:
            gold = exact_query(cache, int(row), mode == "polar_excluded")
            cached = results[mode][1][row]
            difference = max(
                (abs(float(cached[i]) - s) for i, (_, s) in enumerate(gold)), default=0
            )
            benchmark.append({"mode": mode, "count": len(gold), "maximum_score_error": difference})
    if any(b["maximum_score_error"] > 0.05 for b in benchmark):
        raise ValueError(
            "Train bank ANN benchmark has material score error; inspect before training"
        )
    report = {
        **bank,
        "status": "complete",
        "candidate_groups": depth,
        "approximate_retrieval": True,
        "search_calibration": calibration,
        "expansion_search": calibration["selected"]["expansion_search"],
        "exact_full_train_bank_benchmark": benchmark,
        "exact_low_count_confirmations": exact_confirmations,
        "ann_note": "Exact RDKit rerank within ANN candidates; not exhaustive Top10 proof",
        "splits": {},
        "elapsed_this_run_seconds": time.monotonic() - started,
    }
    for mode, (rows, scores) in results.items():
        by_split = {}
        for i, name in enumerate(SPLITS):
            selected = np.flatnonzero(cache.splits == i)
            counts = (rows[selected] >= 0).sum(-1)
            sims = np.asarray(scores[selected])
            usable = counts > 0
            top1 = np.nan_to_num(sims[:, 0], nan=0)
            mean = np.nansum(sims, axis=1) / np.maximum(counts, 1)
            summary = summarize_metrics(counts, top1, mean)
            same, total = 0, 0
            for begin in range(0, len(selected), 8192):
                rr = selected[begin : begin + 8192]
                refs = rows[rr]
                mask = refs >= 0
                same += int(
                    np.count_nonzero(
                        (cache.heads[np.maximum(refs, 0)] == cache.heads[rr, None]) & mask
                    )
                )
                total += int(mask.sum())
            summary.update(
                {
                    "same_polar_template_ratio": same / total if total else None,
                    "zero_template_targets": int(np.count_nonzero(~usable)),
                    "targets_evaluated_including_zero": len(selected),
                }
            )
            by_split[name] = summary
        manifest = {
            **bank,
            "mode": mode,
            "expansion_search": calibration["selected"]["expansion_search"],
            "candidate_groups": depth,
            "search_calibration_sha256": file_hash(work / "train_search_calibration.json"),
            "exclusion": "self/equivalent full graph"
            + (" + canonical polar graph including aliases" if mode == "polar_excluded" else ""),
            "rows_sha256": file_hash(prepared / f"retrieval_{mode}/template_rows.npy"),
            "scores_sha256": file_hash(prepared / f"retrieval_{mode}/scores.npy"),
            "ids_resolved_by": "frozen v0.3 row_ids.npy SHA-bound row order",
        }
        dump_json(prepared / f"retrieval_{mode}/manifest.json", manifest)
        report["splits"][mode] = by_split
    dump_json(report_path, report)
    return report


def formula_stats(prepared):
    """Streaming equal 50/50 mode mixture, using only TRAIN targets + TRAIN references."""
    import torch

    from structure_pretrain.utils.formula import FormulaStandardizer

    torch.set_num_threads(2)
    prepared = Path(prepared)
    report = read_json(prepared / "retrieval_coverage_by_split.json")
    if report["status"] != "complete":
        raise ValueError("Legal retrieval coverage gate not passed")
    splits = load_array(prepared / "row_splits.npy")
    formulas = load_array(prepared / "formulas.npy")
    count = 0

    def examples():
        nonlocal count
        for mode in ("normal", "polar_excluded"):
            rows = load_array(prepared / f"retrieval_{mode}/template_rows.npy")
            training = np.flatnonzero(splits == 0)
            for start in range(0, len(training), 4096):
                selected = training[start : start + 4096]
                refs = np.asarray(rows[selected])
                mask = refs >= 0
                if np.any(splits[refs[mask]] != 0):
                    raise ValueError("Validation/test pair entered formula statistics")
                template = np.asarray(formulas[np.maximum(refs, 0)], dtype=np.float32)
                delta = np.asarray(formulas[selected], dtype=np.float32)[:, None] - template
                count += int(mask.sum())
                yield {
                    "template_mask": torch.from_numpy(mask),
                    "template_formulas": torch.from_numpy(template),
                    "delta_formulas": torch.from_numpy(delta),
                }

    stats = FormulaStandardizer.fit(examples())
    stats.update(
        {
            "schema": SCHEMA,
            "version": "v0.3",
            "structures_sha256": report["structures_sha256"],
            "bank_index_sha256": report["index_sha256"],
            "split_sha256": report["row_splits_sha256"],
            "training_pairs_only": True,
            "mode_mixture": {"normal": 0.5, "polar_excluded": 0.5},
        }
    )
    if stats["pair_count"] != count:
        raise ValueError("Streaming formula count mismatch")
    dump_json(prepared / "formula_stats_train_only.json", stats)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="data/private/pretraining_v03")
    p.add_argument("--frozen", default="data/private/morgan_index_v0.3")
    p.add_argument("--work", default="data/private/pretraining_v03/train_bank")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    retrieve(a.prepared, a.frozen, a.work, a.threads)
    formula_stats(a.prepared)


if __name__ == "__main__":
    main()
