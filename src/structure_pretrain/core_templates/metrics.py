"""Target-denominator metrics for raw beam order and posterior formula filtering."""

import numpy as np

from structure_pretrain.core_templates.chemistry import similarity, validate_candidate


def fg_f1(a, b):
    size = int(a).bit_count() + int(b).bit_count()
    return 2 * (int(a) & int(b)).bit_count() / size if size else 1.0


def score_target(target, candidates, best_template_similarity):
    gold = validate_candidate(target["target_smiles"], target["core_formula"])
    if not gold["valid"] or not gold["formula_match"]:
        raise ValueError("Frozen target validation failed; do not repair source labels")
    records, accepted = [], []
    seen = set()
    for rank, candidate in enumerate(candidates, 1):
        check = (
            validate_candidate(candidate["smiles"], target["core_formula"])
            if candidate["eos"]
            else {"valid": False, "reason": "no_EOS"}
        )
        valid = bool(check["valid"])
        formula_match = valid and bool(check["formula_match"])
        exact = valid and check["anchored_graph_key"] == gold["anchored_graph_key"]
        sim = similarity(check["fingerprint"], gold["fingerprint"]) if valid else 0.0
        fg = fg_f1(check["functional_mask"], gold["functional_mask"]) if valid else 0.0
        record = {
            **candidate,
            "raw_rank": rank,
            "chemical_valid": valid,
            "formula_match": formula_match,
            "exact": exact,
            "similarity": sim,
            "functional_group_F1": fg,
            "rejection": check.get("reason"),
        }
        records.append(record)
        if formula_match and check["anchored_graph_key"] not in seen:
            seen.add(check["anchored_graph_key"])
            accepted.append(record)
    raw, filtered = {}, {}
    for result, choices in ((raw, records), (filtered, accepted)):
        for k in (1, 5, 10):
            result[f"exact_at_{k}"] = int(any(r["exact"] for r in choices[:k]))
        top = choices[:10]
        result["Similarity_at_10"] = max((r["similarity"] for r in top), default=0.0)
        result["functional_group_F1_at_10"] = max(
            (r["functional_group_F1"] for r in top), default=0.0
        )
        result["gain_over_best_input_template"] = (
            result["Similarity_at_10"] - best_template_similarity
        )
        result["beats_best_input_template"] = int(result["gain_over_best_input_template"] > 1e-8)
    raw.update(
        candidate_count=len(records),
        valid_candidates=sum(r["chemical_valid"] for r in records),
        formula_correct_candidates=sum(r["formula_match"] for r in records),
        formula_correct_structure_wrong_candidates=sum(
            r["formula_match"] and not r["exact"] for r in records
        ),
    )
    filtered.update(
        distinct_legal_formula_candidates=len(accepted), no_qualified_candidate=int(not accepted)
    )
    return {
        "target_id": target["target_id"],
        "raw": raw,
        "posterior_filtered": filtered,
        "best_input_template_similarity": float(best_template_similarity),
        "candidates": records,
    }


def aggregate(results):
    """Empty/invalid target outcomes are retained with zero recovery/similarity."""
    if not results:
        raise ValueError("Cannot report an empty evaluation cohort")
    report = {"targets": len(results), "all_targets_in_denominator": True}
    for name in ("raw", "posterior_filtered"):
        keys = results[0][name].keys()
        report[name] = {key: float(np.mean([r[name][key] for r in results])) for key in keys}
    n = sum(r["raw"]["candidate_count"] for r in results)
    f = sum(r["raw"]["formula_correct_candidates"] for r in results)
    report["raw"]["candidate_chemical_valid_fraction"] = sum(
        r["raw"]["valid_candidates"] for r in results
    ) / max(n, 1)
    report["raw"]["formula_correct_but_wrong_structure_fraction"] = sum(
        r["raw"]["formula_correct_structure_wrong_candidates"] for r in results
    ) / max(f, 1)
    return report
