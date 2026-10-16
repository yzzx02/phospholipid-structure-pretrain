import hashlib
from collections import Counter, defaultdict

TIERS = ("CORE", "EXTENDED", "HOLD", "REJECT")


def rank(row, seed):
    return hashlib.sha256(f"{seed}|{row['polar_id']}".encode()).hexdigest()


def stratified_examples(rows, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["polar_family"]].append(row)
    for group in groups.values():
        group.sort(key=lambda row: rank(row, seed))
    result, index = [], 0
    while len(result) < count:
        added = False
        for family in sorted(groups):
            if index < len(groups[family]) and len(result) < count:
                result.append(groups[family][index])
                added = True
        if not added:
            break
        index += 1
    return result


def summarize(rows, controls, seed):
    counts = Counter(row["tier"] for row in rows)
    families = defaultdict(Counter)
    rejected, held = Counter(), Counter()
    for row in rows:
        families[row["polar_family"]][row["tier"]] += 1
        if row["tier"] == "REJECT":
            rejected.update(row["tier_reasons"])
        if row["tier"] == "HOLD":
            held.update(row["tier_reasons"])
    approved = counts["CORE"] + counts["EXTENDED"]
    warnings = [
        f"FAMILY_RETENTION_WARNING:{family}"
        for family, tiers in sorted(families.items())
        if (tiers["HOLD"] + tiers["REJECT"]) / sum(tiers.values()) > 0.9
    ]
    if approved * 7 > 1_500_000:
        warnings.append("ANCHOR_BUDGET_CONFLICT")
    return {
        "input_approvable_count": len(rows),
        "tier_counts": {tier: counts[tier] for tier in TIERS},
        "reactive_hard_reject_count": sum(bool(row["reactive_flags"]) for row in rows),
        "rule_exception_count": sum(bool(row["evaluation_error"]) for row in rows),
        "builder_boundary_hold_count": sum(row.get("builder_boundary_hold", False) for row in rows),
        "family_tier_counts": {
            family: {tier: tiers[tier] for tier in TIERS}
            for family, tiers in sorted(families.items())
        },
        "known_controls": controls,
        "reject_reasons": dict(sorted(rejected.items())),
        "hold_reasons": dict(sorted(held.items())),
        "approved_polar_count": approved,
        "warnings": warnings,
        "budget_estimate": {
            "H": approved,
            "anchor_count": 7,
            "max_structures": 1_500_000,
            "mandatory_anchor_structures": approved * 7,
            "remaining_budget": 1_500_000 - approved * 7,
            "mean_extra_chain_combinations_per_head": (
                (1_500_000 - approved * 7) / approved if approved else None
            ),
        },
        "seed": seed,
        "training_performed": False,
        "full_lipid_library_generated": False,
        "network_requests": 0,
        "literature_or_title_used_for_filtering": False,
        "descriptor_definition": (
            "Open P-boundary fragment; dummy has no mass/heavy-atom count. "
            "Aliphatic run is the longest non-ring, non-aromatic carbon path, "
            "including carbonyl and unsaturated carbons."
        ),
    }
