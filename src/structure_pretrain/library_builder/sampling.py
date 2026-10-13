import hashlib
from collections import defaultdict

from .chain_combinations import permitted
from .schemas import BuildConfig, ChainCombination, PolarRule


def stable_number(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


class StratifiedSampler:
    """Build strata once; head-specific rotations avoid scanning the full bank per head."""

    def __init__(self, bank: list[ChainCombination], config: BuildConfig, exclude_ids=()):
        self.config = config
        self.groups = defaultdict(list)
        excluded = set(exclude_ids)
        bins = config.chain_sampling
        for combo in bank:
            if combo.combination_id in excluded:
                continue
            length = (
                "short"
                if combo.total_carbon <= bins.short_max_total_c
                else "medium"
                if combo.total_carbon <= bins.medium_max_total_c
                else "long"
            )
            unsaturation = (
                "low"
                if combo.total_double_bonds <= bins.low_max_total_db
                else "medium"
                if combo.total_double_bonds <= bins.medium_max_total_db
                else "high"
            )
            # Unsaturation tiers apply to A/A; ether families retain their own strata.
            tier = unsaturation if combo.linkage_family == "AA" else "all"
            self.groups[(combo.linkage_family, length, tier)].append(combo)
        for rows in self.groups.values():
            rows.sort(
                key=lambda combo: (
                    stable_number(f"{config.seed}|{combo.combination_id}"),
                    combo.combination_id,
                )
            )

    def candidates(self, polar: PolarRule, lyso: bool):
        families = defaultdict(list)
        for key in sorted(self.groups):
            rows = self.groups[key]
            if rows[0].is_lyso == lyso and permitted(rows[0], polar, self.config):
                families[key[0]].append((key, rows))
        # Round-robin over AA / OA / PA, then each family's length/unsaturation bins.
        # This prevents a short quota from consuming only the first AA strata.
        positions, remaining = {}, {}
        for groups in families.values():
            for key, rows in groups:
                positions[key] = stable_number(f"{self.config.seed}|{polar.polar_id}|{key}") % len(
                    rows
                )
                remaining[key] = len(rows)
        cursors = {family: 0 for family in families}
        while any(remaining.values()):
            changed = False
            for family in sorted(families):
                groups = families[family]
                for _ in groups:
                    index = cursors[family] % len(groups)
                    cursors[family] += 1
                    key, rows = groups[index]
                    if remaining[key]:
                        yield rows[positions[key] % len(rows)]
                        positions[key] += 1
                        remaining[key] -= 1
                        changed = True
                        break
            if not changed:
                break

    def lyso_candidates(self, polar: PolarRule):
        emitted = set()
        by_token = {
            combo.chain_tokens[0]: combo
            for rows in self.groups.values()
            if rows[0].is_lyso and permitted(rows[0], polar, self.config)
            for combo in rows
        }
        for token in self.config.anchor_single_chains:
            if token in by_token:
                emitted.add(by_token[token].combination_id)
                yield by_token[token]
        for combo in self.candidates(polar, lyso=True):
            if combo.combination_id not in emitted:
                yield combo
