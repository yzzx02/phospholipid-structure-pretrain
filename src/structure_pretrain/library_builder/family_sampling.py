"""Exact integer extra quotas and deterministic distinct chain schedules; no full graphs."""

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import floor, isfinite

from .chain_combinations import permitted
from .sampling import StratifiedSampler, stable_number

FAMILIES = ("phosphate", "phosphonate", "polyphosphate")


@dataclass(frozen=True)
class FamilyBudget:
    mandatory_count: int
    extra_quotas: dict[str, int]
    lyso_quotas: dict[str, int]
    extra_family_totals: dict[str, int]
    unused_budget: int


def weighted_capped_totals(total, counts, weights, cap_per_head, seed):
    if any(f not in FAMILIES for f in counts):
        raise ValueError("Unknown chemistry family in family-aware sampling")
    if any(f not in weights or not isfinite(weights[f]) or weights[f] <= 0 for f in counts):
        raise ValueError("All represented chemistry families need finite positive weights")
    allocations = {f: 0 for f in counts}
    remaining = min(total, sum(counts.values()) * cap_per_head)
    pending = {f for f, count in counts.items() if count}
    while remaining and pending:
        denominator = sum(counts[f] * weights[f] for f in pending)
        targets = {f: remaining * counts[f] * weights[f] / denominator for f in pending}
        capped = {f for f in pending if targets[f] >= counts[f] * cap_per_head}
        if capped:
            for f in capped:
                allocations[f] = counts[f] * cap_per_head
                remaining -= allocations[f]
            pending -= capped
            continue
        bases = {f: floor(targets[f]) for f in pending}
        residual = remaining - sum(bases.values())
        order = sorted(
            pending,
            key=lambda f: (-(targets[f] - bases[f]), stable_number(f"{seed}|family_remainder|{f}")),
        )
        for f in order[:residual]:
            bases[f] += 1
        allocations.update(bases)
        remaining = 0
    return allocations


def allocate_family_budget(polars, details, selected, config, weights):
    config.validate()
    mandatory = sum(len(v) for v in selected.values())
    if mandatory > config.max_structures:
        raise ValueError("Mandatory chain backgrounds exceed configured budget")
    groups = defaultdict(list)
    for p in polars:
        groups[details[p.polar_id]["chemistry_family"]].append(p.polar_id)
    totals = weighted_capped_totals(
        config.max_structures - mandatory,
        {f: len(ids) for f, ids in groups.items()},
        weights,
        config.max_extra_per_head,
        config.seed,
    )
    quotas = {}
    for family, identifiers in groups.items():
        identifiers.sort(key=lambda identifier: stable_number(f"{config.seed}|extras|{identifier}"))
        base, remainder = divmod(totals[family], len(identifiers))
        quotas.update(
            {identifier: base + (i < remainder) for i, identifier in enumerate(identifiers)}
        )
    total_extra = sum(quotas.values())
    lyso_target = min(total_extra, round((mandatory + total_extra) * config.lyso_fraction_target))
    lyso = {
        identifier: (quota * lyso_target // total_extra if total_extra else 0)
        for identifier, quota in quotas.items()
    }
    residual = lyso_target - sum(lyso.values())
    order = sorted(
        quotas,
        key=lambda identifier: (
            -(quotas[identifier] * lyso_target % total_extra) if total_extra else 0,
            stable_number(f"{config.seed}|lyso|{identifier}"),
        ),
    )
    for identifier in order[:residual]:
        lyso[identifier] += 1
    return FamilyBudget(
        mandatory, quotas, lyso, totals, config.max_structures - mandatory - total_extra
    )


def planned_extras(polar, mandatory_ids, budget, sampler):
    remaining = budget.extra_quotas[polar.polar_id]
    seen = set(mandatory_ids)
    lyso_goal = budget.lyso_quotas[polar.polar_id]
    used = 0
    for combo in sampler.lyso_candidates(polar):
        if used >= lyso_goal:
            break
        if combo.combination_id not in seen:
            seen.add(combo.combination_id)
            used += 1
            remaining -= 1
            yield combo
    if used != lyso_goal:
        raise ValueError(f"Insufficient distinct permitted lyso backgrounds for {polar.polar_id}")
    for combo in sampler.candidates(polar, lyso=False):
        if not remaining:
            break
        if combo.combination_id not in seen:
            seen.add(combo.combination_id)
            remaining -= 1
            yield combo
    if remaining:
        raise ValueError(f"Insufficient distinct permitted backgrounds for {polar.polar_id}")


def dry_run_family_schedule(polars, details, selected, bank, config, weights, log=print):
    budget = allocate_family_budget(polars, details, selected, config, weights)
    sampler = StratifiedSampler(bank, config)
    by_id = {c.combination_id: c for c in bank}
    families, linkage, origins, tokens = Counter(), Counter(), Counter(), Counter()
    head_counts, mandatory_family, special_head_counts = {}, Counter(), {}
    schedule_hash = hashlib.sha256()
    selected_count = 0
    for index, polar in enumerate(sorted(polars, key=lambda p: p.polar_id), 1):
        identifier = polar.polar_id
        family = details[identifier]["chemistry_family"]
        actual = 0
        mandatory_family[family] += len(selected[identifier])

        def schedule():
            for label in sorted(selected[identifier]):
                yield by_id[label]
            yield from planned_extras(polar, selected[identifier], budget, sampler)

        for combo in schedule():
            if not permitted(combo, polar, config):
                raise ValueError("Dry-run contains an impermissible chain family")
            actual += 1
            families[family] += 1
            linkage["lyso" if combo.is_lyso else combo.linkage_family] += 1
            origins[details[identifier].get("structure_origin", "unspecified")] += 1
            tokens.update(set(combo.chain_tokens))
            schedule_hash.update(f"{identifier}\t{combo.combination_id}\n".encode())
        head_counts[identifier] = actual
        if details[identifier]["is_special_case"]:
            special_head_counts[identifier] = actual
        selected_count += actual
        if index % 10000 == 0:
            log(
                f"chain-only dry-run {config.max_structures}: {index}/{len(polars)} heads",
                flush=True,
            )
    if selected_count > config.max_structures or any(n < 3 for n in head_counts.values()):
        raise ValueError("Dry-run budget or common-anchor coverage failed")
    return (
        {
            "status": "dry_run_complete_no_full_graphs",
            "seed": config.seed,
            "max_structures": config.max_structures,
            "polar_count": len(polars),
            "polar_family_counts": dict(
                Counter(details[p.polar_id]["chemistry_family"] for p in polars)
            ),
            "mandatory_anchor_and_special_structures": budget.mandatory_count,
            "mandatory_by_chemistry_family": dict(mandatory_family),
            "remaining_budget_after_mandatory": config.max_structures - budget.mandatory_count,
            "estimated_full_structures": selected_count,
            "estimated_full_by_chemistry_family": dict(families),
            "estimated_family_percent": {f: n / selected_count * 100 for f, n in families.items()},
            "estimated_AA": linkage["AA"],
            "estimated_OA": linkage["OA"],
            "estimated_PE_PA": linkage["PA"],
            "estimated_lyso": linkage["lyso"],
            "extra_family_totals": budget.extra_family_totals,
            "family_sampling_weight": weights,
            "per_head_chain_backgrounds_min": min(head_counts.values()),
            "per_head_chain_backgrounds_max": max(head_counts.values()),
            "per_head_chain_backgrounds_mean": selected_count / len(polars),
            "mean_extra_per_head_by_family": {
                f: budget.extra_family_totals[f]
                / sum(details[p.polar_id]["chemistry_family"] == f for p in polars)
                for f in families
            },
            "special_head_background_counts": special_head_counts,
            "all_special_heads_retained": all(v >= 3 for v in special_head_counts.values()),
            "estimated_structures_by_origin_provenance_only": dict(origins),
            "schedule_sha256": schedule_hash.hexdigest(),
            "within_configured_budget": selected_count <= config.max_structures,
            "within_2000000": selected_count <= 2_000_000,
            "unused_budget": config.max_structures - selected_count,
            "distinct_chain_ids_per_head": True,
            "chain_token_counts": dict(sorted(tokens.items())),
            "full_graph_canonical_dedup_performed": False,
            "estimate_limitation": (
                "Counts are exact for planned head/chain pairs. Full assembled-graph canonical "
                "dedup and assembly rejection can reduce realized counts."
            ),
            "training_performed": False,
            "full_lipid_library_generated": False,
        },
        budget,
        head_counts,
    )
