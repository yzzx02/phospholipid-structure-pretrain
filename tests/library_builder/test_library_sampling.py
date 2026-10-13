from itertools import islice

from structure_pretrain.library_builder.chain_combinations import (
    active_anchors,
    build_combination_bank,
)
from structure_pretrain.library_builder.sampling import StratifiedSampler


def test_stratified_extras_cover_available_linkage_length_and_unsaturation(inputs):
    chains, polars, config = inputs
    anchors = active_anchors(chains, config, [])
    sampler = StratifiedSampler(
        build_combination_bank(chains, config), config, [combo.combination_id for combo in anchors]
    )
    chosen = list(islice(sampler.candidates(polars["demo_PE"], lyso=False), 30))
    assert {combo.linkage_family for combo in chosen[:3]} == {"AA", "OA", "PA"}
    aa = [combo for combo in chosen if combo.linkage_family == "AA"]
    lengths = {
        "short" if combo.total_carbon <= 32 else "medium" if combo.total_carbon <= 40 else "long"
        for combo in aa
    }
    db = {
        "low"
        if combo.total_double_bonds <= 2
        else "medium"
        if combo.total_double_bonds <= 4
        else "high"
        for combo in aa
    }
    assert lengths == {"short", "medium", "long"}
    assert db == {"low", "medium", "high"}
