from dataclasses import dataclass

from .schemas import BuildConfig, ChainCombination, PolarRule


class MandatoryBudgetExceeded(ValueError):
    def __init__(self, count: int, maximum: int, anchors_per_head: int):
        super().__init__("Mandatory anchor structures exceed budget.")
        self.mandatory_anchor_count = count
        self.max_allowed_eligible_heads = maximum // anchors_per_head if anchors_per_head else 0


@dataclass(frozen=True)
class BudgetPlan:
    mandatory_anchor_count: int
    remaining_budget: int
    extra_quotas: dict[str, int]
    lyso_quotas: dict[str, int]


def allocate_budget(polars: list[PolarRule], anchors: list[ChainCombination], config: BuildConfig):
    config.validate()
    if not polars:
        raise ValueError("No validated polar moieties")
    mandatory = sum(polar.allow_aa for polar in polars) * len(anchors)
    if mandatory > config.max_structures:
        raise MandatoryBudgetExceeded(mandatory, config.max_structures, len(anchors))
    remaining = config.max_structures - mandatory
    base, remainder = divmod(remaining, len(polars))
    quotas = {
        polar.polar_id: min(config.max_extra_per_head, base + int(index < remainder))
        for index, polar in enumerate(polars)
    }
    projected_total = mandatory + sum(quotas.values())
    lyso_target = min(sum(quotas.values()), round(projected_total * config.lyso_fraction_target))
    lyso = {polar.polar_id: 0 for polar in polars}
    while lyso_target:
        changed = False
        for polar in polars:
            if lyso[polar.polar_id] < quotas[polar.polar_id]:
                lyso[polar.polar_id] += 1
                lyso_target -= 1
                changed = True
                if not lyso_target:
                    break
        if not changed:
            break
    return BudgetPlan(mandatory, remaining, quotas, lyso)
