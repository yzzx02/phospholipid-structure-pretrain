"""Bounded-memory exhaustive binary Tanimoto, deterministic graph-ID tie handling."""

import numpy as np
from rdkit import DataStructs
from usearch.index import search


def bitvectors(packed):
    return [DataStructs.CreateFromBinaryText(bytes(row)) for row in packed]


def oracle(query, bank_fp, bank_keys, query_key, tie_order, k):
    scores = np.asarray(
        DataStructs.BulkTanimotoSimilarity(DataStructs.CreateFromBinaryText(bytes(query)), bank_fp),
        dtype=np.float64,
    )
    scores[bank_keys == query_key] = -1
    order = np.lexsort((tie_order, -scores))
    order = order[scores[order] >= 0][:k]
    return order, scores[order]


def exact_retrieve(
    queries,
    bank,
    query_keys,
    bank_keys,
    bank_ids,
    *,
    k=3,
    batch_size=2048,
    candidate_count=16,
    audit_count=512,
    seed=20261023,
    event=None,
):
    queries = np.ascontiguousarray(queries, dtype=np.uint8)
    bank = np.ascontiguousarray(bank, dtype=np.uint8)
    assert queries.ndim == bank.ndim == 2 and queries.shape[1] == bank.shape[1] == 256
    bank_keys = np.asarray(bank_keys)
    ids = np.asarray(bank_ids, dtype=str)
    assert len(np.unique(ids)) == len(bank)
    tie_order = np.empty(len(bank), dtype=np.int64)
    tie_order[np.argsort(ids, kind="stable")] = np.arange(len(bank))
    bank_fp = bitvectors(bank)
    output = np.full((len(queries), k), -1, dtype=np.int32)
    similarities = np.zeros((len(queries), k), dtype=np.float32)
    fallbacks = 0
    candidates = min(max(candidate_count, k + 1), len(bank))
    for start in range(0, len(queries), batch_size):
        q = queries[start : start + batch_size]
        found = search(
            bank, q, count=candidates, metric="tanimoto", dtype="b1", exact=True, threads=4
        )
        keys = np.atleast_2d(found.keys)
        distances = np.atleast_2d(found.distances)
        counts = (
            np.asarray(found.counts).reshape(-1)
            if hasattr(found, "counts")
            else np.asarray([len(found.keys)])
        )
        for offset in range(len(q)):
            i = start + offset
            choices = keys[offset, : int(counts[offset])].astype(np.int64)
            choices = choices[bank_keys[choices] != query_keys[i]]
            vector = DataStructs.CreateFromBinaryText(bytes(q[offset]))
            values = np.asarray(
                DataStructs.BulkTanimotoSimilarity(vector, [bank_fp[j] for j in choices])
            )
            order = np.lexsort((tie_order[choices], -values))
            selected = order[:k]
            # Native float32 scores may hide equal-score identities at the topK
            # boundary. Exhaustive RDKit resolves every ambiguous boundary.
            boundary = 1 - float(distances[offset, int(counts[offset]) - 1])
            ambiguous = len(selected) < k or (
                len(selected) and values[selected[-1]] - boundary <= 1e-6
            )
            if ambiguous:
                chosen, values = oracle(q[offset], bank_fp, bank_keys, query_keys[i], tie_order, k)
                fallbacks += 1
            else:
                chosen, values = choices[selected], values[selected]
            output[i, : len(chosen)] = chosen
            similarities[i, : len(chosen)] = values
        if event:
            event(
                stage="exact_Tanimoto",
                queries=min(start + len(q), len(queries)),
                total=len(queries),
                bank=len(bank),
                tie_fallbacks=fallbacks,
            )
    sample = np.random.default_rng(seed).choice(
        len(queries), min(audit_count, len(queries)), replace=False
    )
    matches = 0
    for i in sample:
        expected, scores = oracle(queries[i], bank_fp, bank_keys, query_keys[i], tie_order, k)
        actual = output[i][output[i] >= 0]
        matches += int(
            np.array_equal(actual, expected)
            and np.allclose(similarities[i, : len(expected)], scores, rtol=0, atol=1e-6)
        )
    audit = {
        "method": (
            "exhaustive USearch binary Tanimoto + exact RDKit reranking and boundary tie fallback"
        ),
        "queries": len(queries),
        "bank": len(bank),
        "K": k,
        "oracle_sample": len(sample),
        "oracle_identity_and_score_matches": matches,
        "oracle_required_fraction": 1.0,
        "oracle_match_fraction": matches / max(len(sample), 1),
        "boundary_tie_fallbacks": fallbacks,
        "tie_policy": "score descending, lexicographic template ID ascending",
        "all_pairs_matrix_materialized": False,
        "maximum_query_batch": batch_size,
    }
    if matches != len(sample):
        raise ValueError("Exhaustive retrieval/RDKit identity or score audit failed: " + str(audit))
    return output, similarities, audit
