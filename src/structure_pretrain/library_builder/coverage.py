"""Count chain exposure; P coverage is descriptive under the user's updated policy."""


def chain_coverage(chains, counts):
    families = {}
    for family in "AOP":
        expected = sorted(t for t, c in chains.items() if c.linkage_type == family)
        minimum = 0 if family == "P" else 1
        insufficient = [t for t in expected if counts.get(t, 0) < minimum]
        families[family] = {
            "expected": len(expected),
            "covered": sum(counts.get(t, 0) > 0 for t in expected),
            "minimum_required": minimum,
            "coverage_is_hard_requirement": family != "P",
            "min_count": min((counts.get(t, 0) for t in expected), default=0),
            "max_count": max((counts.get(t, 0) for t in expected), default=0),
            "insufficient_tokens": insufficient,
        }
    return {
        "status": "passed"
        if all(not f["insufficient_tokens"] for f in families.values())
        else "failed",
        "definition": "Distinct full graphs/pairs containing a token; counted once per graph",
        "p_policy": "P/A only for exact PE; no P-coverage reservation or hard gate; non-PE uses O",
        "families": families,
        "chain_token_counts": {t: counts.get(t, 0) for t in sorted(chains)},
    }
