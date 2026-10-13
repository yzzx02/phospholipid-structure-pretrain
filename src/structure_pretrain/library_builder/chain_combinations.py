from itertools import combinations_with_replacement, product

from .chain_library import chain_sort_key
from .schemas import BuildConfig, ChainCombination, ChainRule, PolarRule, Rejection


def make_combination(tokens: tuple[str, ...], chains: dict[str, ChainRule], mapping=None):
    ordered = tuple(sorted(tokens, key=chain_sort_key))
    types = "".join(chains[token].linkage_type for token in ordered)
    if types not in {"AA", "OA", "PA", "A", "O", "P"}:
        raise ValueError("Unsupported chain combination")
    inverse = {
        token
        for token, mapped in (mapping or {}).items()
        if mapped in ordered and types in {"O", "OA"}
    }
    return ChainCombination(
        "_".join(ordered),
        ordered,
        types,
        sum(chains[t].carbon_count for t in ordered),
        sum(chains[t].double_bond_count for t in ordered),
        len(ordered) == 1,
        tuple(sorted(inverse)),
    )


def build_combination_bank(chains: dict[str, ChainRule], config: BuildConfig, mapping=None):
    groups = {
        linkage: sorted(
            (t for t, c in chains.items() if c.linkage_type == linkage), key=chain_sort_key
        )
        for linkage in "AOP"
    }
    a, o, p = groups["A"], groups["O"], groups["P"]
    count = len(a) * (len(a) + 1) // 2 + len(o) * len(a) + len(a) + len(o)
    if config.enable_pe_p:
        count += len(p) * len(a)
    if config.include_p_lyso:
        count += len(p)
    if count > config.max_chain_combinations:
        raise ValueError(
            "Chain combination bank exceeds configured limit; no head Cartesian product started"
        )
    tuples = list(combinations_with_replacement(a, 2)) + list(product(o, a))
    if config.enable_pe_p:
        tuples += list(product(p, a))
    tuples += [(token,) for token in a + o]
    if config.include_p_lyso:
        tuples += [(token,) for token in p]
    return [make_combination(tokens, chains, mapping) for tokens in tuples]


def permitted(combo: ChainCombination, polar: PolarRule, config: BuildConfig):
    if combo.linkage_family in {"A", "AA"}:
        return polar.allow_aa
    if combo.linkage_family in {"O", "OA"}:
        return polar.allow_oa
    return (
        polar.polar_family == "PE"
        and polar.allow_pa
        and config.enable_pe_p
        and (not combo.is_lyso or config.include_p_lyso)
    )


def active_anchors(chains, config, rejections: list[Rejection]):
    anchors = {}
    for label in config.anchor_combinations:
        tokens = tuple(label.split("_"))
        if len(tokens) != 2 or any(not token.startswith("A-") for token in tokens):
            raise ValueError("Anchor combinations must be A/A pairs")
        if any(token not in chains for token in tokens):
            rejections.append(
                Rejection(
                    "anchors",
                    label,
                    "anchor_chain_missing",
                    "Anchor disabled because an external chain token is missing",
                )
            )
            continue
        combo = make_combination(tokens, chains)
        anchors[combo.combination_id] = combo
    return list(anchors.values())
