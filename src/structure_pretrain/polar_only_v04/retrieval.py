"""Own TRAIN-only chain/head postings and NEW chainless-Morgan retrieval.

Offline queries know the target graph. This is pretraining retrieval, not a
spectral query implementation or an MS/MS identification accuracy benchmark.
"""

from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import DataStructs

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04 import SCHEMA
from structure_pretrain.polar_only_v04.prepare import V03, dump, emit
from structure_pretrain.retrieval.indexed_morgan import make_index, search_many

K = 10
THRESHOLD = 0.2


def chosen_mode(rows, h_count, hc_count, fraction=0.8, seed=42, force_h=False):
    """Stable target/group draws, explicit rare-chain fallback, never drop a row."""
    rows = np.asarray(rows, dtype=np.int64)
    draw = ((rows * 2654435761 + seed * 2246822519) & 0xFFFFFFFF) / 2**32
    desired_h = np.ones(len(rows), bool) if force_h else draw < fraction
    h = desired_h.copy()
    h[desired_h & (h_count == 0)] = False
    h[~desired_h & (hc_count == 0)] = True
    if np.any(np.where(h, h_count, hc_count) == 0):
        raise ValueError("Zero legal templates after explicit H/HC fallback; stop training")
    return h, h != desired_h


def exact_scores(query, bank):
    intersection = np.bitwise_count(np.bitwise_and(bank, query)).sum(-1)
    union = np.bitwise_count(np.bitwise_or(bank, query)).sum(-1)
    return intersection / np.maximum(union, 1)


def quantiles(values):
    values = np.asarray(values)
    return np.quantile(values, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).tolist() if len(values) else []


def statistics_path(root, k=3, h_fraction=0.8, seed=42):
    suffix = "" if (k, h_fraction, seed) == (3, 0.8, 42) else f"_K{k}_H{h_fraction:g}_s{seed}"
    return Path(root) / f"formula_stats_train_only{suffix}.json"


def prepare_retrieval(root):
    root = Path(root)
    if (root / "retrieval_audit.json").exists():
        assert read_json(root / "retrieval_audit.json")["status"] == "passed"
        if not statistics_path(root).exists():
            fit_statistics(root)
        return
    assert read_json(root / "formula_audit.json")["status"] == "passed"
    features = read_json(root / "features_manifest.json")
    groups = np.load(V03 / "polar_groups.npy", mmap_mode="r")
    split = np.load(V03 / "row_splits.npy", mmap_mode="r")
    labels = np.load(V03 / "chain_labels.npy", mmap_mode="r")
    code = labels[:, 0].astype(np.int64) * 323 + labels[:, 1]
    variants = np.load(root / "row_variants.npy", mmap_mode="r")
    heads = np.load(root / "variant_heads.npy", mmap_mode="r")
    fp = np.load(root / "chainless_fingerprints.npy", mmap_mode="r")
    n = len(groups)
    train = np.flatnonzero(split == 0)
    # Actual chain-combination postings and variant/head postings, TRAIN rows only.
    order = train[np.lexsort((train, variants[train], code[train]))]
    combo_counts = np.bincount(code[train], minlength=323**2)
    combo_offsets = np.r_[0, combo_counts.cumsum()]
    np.save(root / "chain_posting_rows.npy", order.astype(np.int32))
    np.save(root / "chain_posting_offsets.npy", combo_offsets)
    vo = train[np.lexsort((train, code[train], variants[train]))]
    vc = np.bincount(variants[train], minlength=len(fp))
    offsets = np.r_[0, vc.cumsum()]
    np.save(root / "head_variant_posting_rows.npy", vo.astype(np.int32))
    np.save(root / "head_variant_posting_offsets.npy", offsets)
    first = np.full(len(fp), -1, np.int32)
    second = first.copy()
    bank = np.flatnonzero(vc)
    for v in bank:
        rows = vo[offsets[v] : offsets[v + 1]]
        first[v] = rows[0]
        other = rows[code[rows] != code[rows[0]]]
        if len(other):
            second[v] = other[0]
    np.save(root / "variant_first_train_row.npy", first)
    np.save(root / "variant_other_chain_train_row.npy", second)
    bank_path = root / "train_chainless_morgan.usearch"
    emit({"stage": "new_train_chainless_index", "variants": len(bank)})
    from usearch.index import Index

    if bank_path.exists():
        index = Index.restore(str(bank_path))
    else:
        index = make_index(fp[bank], keys=bank.astype(np.uint64), threads=4)
        index.save(str(bank_path))
    width = 128
    candidate_path = root / "ann_candidates.npy"
    similarity_path = root / "ann_similarities.npy"
    if not (root / "ann_complete.json").exists():
        candidates = np.lib.format.open_memmap(
            candidate_path, mode="w+", dtype=np.int32, shape=(len(fp), width)
        )
        sims = np.lib.format.open_memmap(
            similarity_path, mode="w+", dtype=np.float32, shape=(len(fp), width)
        )
        for start in range(0, len(fp), 512):
            end = min(start + 512, len(fp))
            found = search_many(index, fp[start:end], count=width, threads=4)
            assert np.all(found.counts == width)
            keys = found.keys.astype(np.int32)
            score = exact_scores(fp[start:end, None, :], fp[keys])
            rank = np.lexsort((keys, -score), axis=1)
            candidates[start:end] = np.take_along_axis(keys, rank, 1)
            sims[start:end] = np.take_along_axis(score, rank, 1)
            if start % 8192 == 0:
                emit({"stage": "chainless_ann_queries_exact_rerank", "variants": end})
        candidates.flush()
        sims.flush()
        dump(root / "ann_complete.json", {"queries": len(fp), "width": width})
    candidates = np.load(candidate_path, mmap_mode="r")
    sims = np.load(similarity_path, mmap_mode="r")
    # A common lyso chain can make most nearest variants ineligible for HC.
    # Index the eligible TRAIN bank BEFORE retrieval, rather than repeatedly
    # scanning all351k variants or filtering an insufficient global shortlist.
    hc_special = {}
    for combo in np.flatnonzero(combo_counts > 4096):
        available = first[bank].copy()
        same = code[available] == combo
        available[same] = second[bank[same]]
        if np.mean(available < 0) <= 0.1:
            continue
        eligible_bank = bank[available >= 0]
        query_ids = np.unique(variants[code == combo])
        stem = f"HC_eligible_{combo}"
        marker = root / f"{stem}_complete.json"
        candidate_file, score_file = root / f"{stem}_candidates.npy", root / f"{stem}_scores.npy"
        special_index_path = root / f"{stem}.usearch"
        if not marker.exists():
            emit(
                {
                    "stage": "HC_chain_excluded_bank_index",
                    "combo": int(combo),
                    "eligible_train_variants": len(eligible_bank),
                    "queries": len(query_ids),
                }
            )
            if special_index_path.exists():
                special_index = Index.restore(str(special_index_path))
            else:
                special_index = make_index(
                    fp[eligible_bank], keys=eligible_bank.astype(np.uint64), threads=4
                )
                special_index.save(str(special_index_path))
            ca = np.lib.format.open_memmap(
                candidate_file, mode="w+", dtype=np.int32, shape=(len(query_ids), width)
            )
            ss = np.lib.format.open_memmap(
                score_file, mode="w+", dtype=np.float32, shape=(len(query_ids), width)
            )
            for start in range(0, len(query_ids), 512):
                ids = query_ids[start : start + 512]
                found = search_many(special_index, fp[ids], count=width, threads=4)
                keys = found.keys.astype(np.int32)
                score = exact_scores(fp[ids, None, :], fp[keys])
                rank = np.lexsort((keys, -score), axis=1)
                ca[start : start + len(ids)] = np.take_along_axis(keys, rank, 1)
                ss[start : start + len(ids)] = np.take_along_axis(score, rank, 1)
                if start % 8192 == 0:
                    emit({"stage": "HC_chain_excluded_queries", "queries": start + len(ids)})
            ca.flush()
            ss.flush()
            dump(
                marker,
                {
                    "chain_code": int(combo),
                    "bank_variants": len(eligible_bank),
                    "query_variants": len(query_ids),
                    "index_sha256": file_hash(special_index_path),
                },
            )
        hc_special[int(combo)] = (
            query_ids,
            np.load(candidate_file, mmap_mode="r"),
            np.load(score_file, mmap_mode="r"),
        )
    arrays = {}
    h_marker = root / "H_build_complete.json"
    reuse_h = h_marker.exists()
    if reuse_h:
        frozen = read_json(h_marker)
        for name in ("H_rows.npy", "H_scores.npy"):
            assert file_hash(root / name) == frozen[name]
    for mode in ("H", "HC"):
        arrays[mode] = (
            np.lib.format.open_memmap(
                root / f"{mode}_rows.npy",
                mode="r+" if mode == "H" and reuse_h else "w+",
                dtype=np.int32,
                shape=(n, K),
            ),
            np.lib.format.open_memmap(
                root / f"{mode}_scores.npy",
                mode="r+" if mode == "H" and reuse_h else "w+",
                dtype=np.float32,
                shape=(n, K),
            ),
        )
        if mode != "H" or not reuse_h:
            arrays[mode][0][:] = -1
            arrays[mode][1][:] = 0
    hrows, hscores = arrays["H"]
    hcrows, hcscores = arrays["HC"]
    allorder = np.argsort(code, kind="stable")
    alloffsets = np.r_[0, np.bincount(code, minlength=323**2).cumsum()]
    exact_fallbacks = Counter()
    # H ranks its OWN exact-chain postings, not a filter of old v0.3 Top10.
    for combo in () if reuse_h else np.flatnonzero(np.diff(alloffsets)):
        targets = allorder[alloffsets[combo] : alloffsets[combo + 1]]
        refs = order[combo_offsets[combo] : combo_offsets[combo + 1]]
        if not len(refs):
            continue
        rv = variants[refs]
        if len(refs) <= 4096:
            vectors = [DataStructs.CreateFromBinaryText(bytes(x)) for x in fp[rv]]
            for row in targets:
                query = DataStructs.CreateFromBinaryText(bytes(fp[variants[row]]))
                score = np.asarray(DataStructs.BulkTanimotoSimilarity(query, vectors))
                legal = (groups[refs] != groups[row]) & (score >= THRESHOLD)
                eligible = np.flatnonzero(legal)
                pick = eligible[np.lexsort((refs[eligible], -score[eligible]))[:K]]
                hrows[row, : len(pick)] = refs[pick]
                hscores[row, : len(pick)] = score[pick]
        else:
            for start in range(0, len(targets), 1024):
                rows = targets[start : start + 1024]
                cand = candidates[variants[rows]]
                scores = sims[variants[rows]]
                pos = np.searchsorted(rv, cand)
                exists = (pos < len(rv)) & (rv[np.minimum(pos, len(rv) - 1)] == cand)
                refrows = refs[np.minimum(pos, len(rv) - 1)]
                legal = exists & (heads[cand] != groups[rows, None]) & (scores >= THRESHOLD)
                for j, row in enumerate(rows):
                    pick = np.flatnonzero(legal[j])[:K]
                    if len(pick) < K:
                        score = exact_scores(fp[variants[row]], fp[rv])
                        ok = np.flatnonzero((heads[rv] != groups[row]) & (score >= THRESHOLD))
                        chosen = ok[np.lexsort((refs[ok], -score[ok]))[:K]]
                        hrows[row, : len(chosen)] = refs[chosen]
                        hscores[row, : len(chosen)] = score[chosen]
                        exact_fallbacks["H"] += 1
                    else:
                        hrows[row] = refrows[j, pick]
                        hscores[row] = scores[j, pick]
        if combo % 500 == 0 or len(refs) > 4096:
            emit({"stage": "same_chain_H", "combo": int(combo), "targets": len(targets)})
    hrows.flush()
    hscores.flush()
    dump(h_marker, {name: file_hash(root / name) for name in ("H_rows.npy", "H_scores.npy")})
    # HC chooses a real TRAIN member of another chain combination, then de-duplicates heads.
    for row in range(n):
        if int(code[row]) in hc_special:
            query_ids, ca, ss = hc_special[int(code[row])]
            position = np.searchsorted(query_ids, variants[row])
            cand, score = ca[position], ss[position]
        else:
            cand, score = candidates[variants[row]], sims[variants[row]]
        picks, values, seen = [], [], set()
        for v, value in zip(cand, score):
            g = int(heads[v])
            if value < THRESHOLD:
                break
            if g == groups[row] or g in seen:
                continue
            ref = int(first[v])
            if code[ref] == code[row]:
                ref = int(second[v])
            if ref < 0:
                continue
            picks.append(ref)
            values.append(float(value))
            seen.add(g)
            if len(picks) == K:
                break
        if len(picks) < K:
            # Exhaustive confirmation prevents an ANN miss being reported as true zero coverage.
            score_all = exact_scores(fp[variants[row]], fp[bank])
            ranked = np.lexsort((bank, -score_all))
            picks, values, seen = [], [], set()
            for position in ranked:
                v, value = int(bank[position]), float(score_all[position])
                if value < THRESHOLD:
                    break
                g = int(heads[v])
                if g == groups[row] or g in seen:
                    continue
                ref = int(first[v])
                if code[ref] == code[row]:
                    ref = int(second[v])
                if ref >= 0:
                    picks.append(ref)
                    values.append(value)
                    seen.add(g)
                if len(picks) == K:
                    break
            exact_fallbacks["HC"] += 1
        hcrows[row, : len(picks)] = picks
        hcscores[row, : len(values)] = values
        if row % 100000 == 0:
            emit({"stage": "different_chain_HC", "rows": row})
    for refs, scores in arrays.values():
        refs.flush()
        scores.flush()
    # Independent complete legality audit plus recomputation of every stored similarity.
    for mode, (refs, scores) in arrays.items():
        for start in range(0, n, 4096):
            end = min(start + 4096, n)
            block = refs[start:end]
            mask = block >= 0
            target = np.broadcast_to(np.arange(start, end)[:, None], block.shape)[mask]
            chosen = block[mask]
            assert np.all(split[chosen] == 0)
            assert np.all(groups[chosen] != groups[target])
            assert np.all(chosen != target)
            assert np.all((code[chosen] == code[target]) == (mode == "H"))
            expected = exact_scores(fp[variants[target]], fp[variants[chosen]])
            np.testing.assert_allclose(scores[start:end][mask], expected, atol=1e-7)
            assert np.all(expected >= THRESHOLD)
    counts = {m: (a[0] >= 0).sum(-1) for m, a in arrays.items()}
    gold_rows = np.random.default_rng(42).choice(np.flatnonzero(split == 1), 64, replace=False)
    gold = {m: [] for m in arrays}
    for j, row in enumerate(gold_rows):
        refs = order[combo_offsets[code[row]] : combo_offsets[code[row] + 1]]
        legal = refs[groups[refs] != groups[row]]
        scores = exact_scores(fp[variants[row]], fp[variants[legal]])
        optimum = float(scores.max()) if len(scores) else 0.0
        optimum = optimum if optimum >= THRESHOLD else 0.0
        gold["H"].append(max(0.0, optimum - float(hscores[row, 0])))
        scores = exact_scores(fp[variants[row]], fp[bank])
        available = first[bank].copy()
        same = code[available] == code[row]
        available[same] = second[bank[same]]
        legal = (available >= 0) & (heads[bank] != groups[row])
        optimum = float(scores[legal].max())
        optimum = optimum if optimum >= THRESHOLD else 0.0
        gold["HC"].append(max(0.0, optimum - float(hcscores[row, 0])))
        if j % 8 == 0:
            emit({"stage": "independent_exhaustive_rank_audit", "targets": j + 1})
    rank_audit = {
        m: {
            "targets": 64,
            "optimal_top1_score_fraction": float(np.mean(np.asarray(gaps) < 1e-7)),
            "top1_score_gap_quantiles": quantiles(gaps),
        }
        for m, gaps in gold.items()
    }
    if any(x["optimal_top1_score_fraction"] < 0.95 for x in rank_audit.values()):
        raise ValueError(f"ANN retrieval fails independent exact top1 score gate: {rank_audit}")
    head_min = np.full(int(groups.max()) + 1, 255, np.uint8)
    head_max = np.zeros(len(head_min), np.uint8)
    np.minimum.at(head_min, groups, split)
    np.maximum.at(head_max, groups, split)
    assert np.array_equal(head_min, head_max), "Polar graph splits intersect"
    report = {}
    for s, name in enumerate(("train", "validation", "test")):
        rows = np.flatnonzero(split == s)
        selected_h, fallback = chosen_mode(rows, counts["H"][rows], counts["HC"][rows])
        selected_count = np.where(selected_h, counts["H"][rows], counts["HC"][rows])
        pairs_h = np.minimum(counts["H"][rows][selected_h], 3).sum()
        pair_total = np.minimum(selected_count, 3).sum()
        report[name] = {
            "targets": len(rows),
            "modes": {
                m: {
                    "at_least": {str(k): int((counts[m][rows] >= k).sum()) for k in (1, 3, 5)},
                    "zero": int((counts[m][rows] == 0).sum()),
                    "less_than_K3": int((counts[m][rows] < 3).sum()),
                    "top1_quantiles": quantiles(a[1][rows, 0][counts[m][rows] > 0]),
                    "topK_quantiles": quantiles(a[1][rows][a[0][rows] >= 0]),
                }
                for m, a in arrays.items()
            },
            "actual_H_target_fraction": float(selected_h.mean()),
            "fallback_targets": int(fallback.sum()),
            "zero_after_fallback": int((selected_count == 0).sum()),
            "same_chain_template_pair_fraction_K3": float(pairs_h / pair_total),
            "different_head_template_pair_fraction": 1.0,
        }
    bank_hash = file_hash(root / "chain_posting_rows.npy")
    dump(
        root / "retrieval_audit.json",
        {
            "schema": SCHEMA,
            "status": "passed",
            "threshold": THRESHOLD,
            "cache_K": K,
            "h_fraction_requested": 0.8,
            "seed": 42,
            "train_bank_rows": len(train),
            "train_bank_variants": len(bank),
            "observed_chain_combinations": int(len(np.unique(code))),
            "train_bank_sha256": bank_hash,
            "new_chainless_features_sha256": features["fingerprints_sha256"],
            "index_sha256": file_hash(bank_path),
            "independent_exhaustive_rank_audit": rank_audit,
            "split_counts": {
                name: int((split == s).sum())
                for s, name in enumerate(("train", "validation", "test"))
            },
            "canonical_polar_graph_split_intersections": 0,
            "exact_fallback_queries": dict(exact_fallbacks),
            "HC_eligible_bank_indexes": {
                str(combo): read_json(root / f"HC_eligible_{combo}_complete.json")
                for combo in hc_special
            },
            "ranking": "H exact postings <=4096; otherwise new ANN128 + exact rerank; HC ANN128",
            "oracle_target_structure_known_offline_only": True,
            "old_full_molecule_fingerprints_used": False,
            "all_stored_references_train_only_different_head_and_similarities_verified": True,
            "splits": report,
            "artifacts_sha256": {
                name: file_hash(root / name)
                for name in ("H_rows.npy", "H_scores.npy", "HC_rows.npy", "HC_scores.npy")
            },
        },
    )
    fit_statistics(root, groups, split, counts)


def fit_statistics(root, groups=None, split=None, counts=None, k=3, h_fraction=0.8, seed=42):
    """Streaming 20D moments from TRAIN-target / TRAIN-template pairs only."""
    groups = np.load(V03 / "polar_groups.npy", mmap_mode="r") if groups is None else groups
    split = np.load(V03 / "row_splits.npy", mmap_mode="r") if split is None else split
    refs = {m: np.load(root / f"{m}_rows.npy", mmap_mode="r") for m in ("H", "HC")}
    counts = {m: (v >= 0).sum(-1) for m, v in refs.items()} if counts is None else counts
    formulas = np.load(root / "head_polar_formulas.npy", mmap_mode="r")
    rows = np.flatnonzero(split == 0)
    total, sums, squares = 0, np.zeros(20), np.zeros(20)
    for start in range(0, len(rows), 4096):
        target = rows[start : start + 4096]
        h, _ = chosen_mode(target, counts["H"][target], counts["HC"][target], h_fraction, seed)
        chosen = np.where(h[:, None], refs["H"][target, :k], refs["HC"][target, :k])
        mask = chosen >= 0
        f = formulas[groups[np.maximum(chosen, 0)]].astype(np.float64)
        delta = formulas[groups[target]][:, None].astype(np.float64) - f
        values = np.concatenate((f, delta), -1)[mask]
        total += len(values)
        sums += values.sum(0)
        squares += (values**2).sum(0)
    mean = sums / total
    std = np.sqrt(np.maximum(squares / total - mean**2, 0))
    std[std < 1e-8] = 1
    dump(
        statistics_path(root, k, h_fraction, seed),
        {
            "training_pairs_only": True,
            "pairs": total,
            "K": k,
            "h_fraction": h_fraction,
            "seed": seed,
            "template_mean": mean[:10].tolist(),
            "template_std": std[:10].tolist(),
            "delta_mean": mean[10:].tolist(),
            "delta_std": std[10:].tolist(),
            "retrieval_audit_sha256": file_hash(root / "retrieval_audit.json"),
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-only", action="store_true")
    parser.add_argument("--k", type=int, choices=(3, 5, 10), default=3)
    parser.add_argument("--h-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.h_fraction <= 1:
        parser.error("H fraction must be within0..1")
    root = Path("data/private/pretraining_v04")
    if args.statistics_only:
        fit_statistics(root, k=args.k, h_fraction=args.h_fraction, seed=args.seed)
    else:
        prepare_retrieval(root)
