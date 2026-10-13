from dataclasses import replace

from structure_pretrain.library_builder.chain_combinations import build_combination_bank, permitted
from structure_pretrain.library_builder.chain_library import import_p_to_o
from structure_pretrain.library_builder.pipeline import build_library

from .conftest import EXAMPLES


def test_only_declared_pe_gets_independent_p(inputs):
    chains, polars, config = inputs
    bank = build_combination_bank(chains, config)
    pa = [combo for combo in bank if combo.linkage_family == "PA"]
    assert pa and all(permitted(combo, polars["demo_PE"], config) for combo in pa)
    for head in ("demo_PC", "demo_expanded", "demo_phosphonate"):
        assert all(not permitted(combo, polars[head], config) for combo in pa)
    assert all(
        not permitted(combo, polars["demo_PE"], replace(config, enable_pe_p=False)) for combo in pa
    )
    assert not any(combo.linkage_family == "P" for combo in bank)


def test_mapping_is_external_and_missing_mapping_is_logged(inputs, tmp_path):
    chains, _, config = inputs
    errors = []
    mapping = import_p_to_o(EXAMPLES / "p_to_o_mapping.tsv", chains, errors)
    assert mapping == {"P-16:0": "O-16:1"} and not errors
    report = build_library(
        config, EXAMPLES / "chains.tsv", EXAMPLES / "polar_moieties.tsv", tmp_path / "missing"
    )
    assert report["rejection_reasons"]["p_to_o_mapping_missing"] == 3
    assert report["PE_PA_count"] > 0


def test_invalid_mapping_does_not_generate_new_chains(inputs, tmp_path):
    chains, _, _ = inputs
    path = tmp_path / "bad.tsv"
    path.write_text("p_token\to_token\nP-16:0\tO-16:0\n", encoding="utf-8")
    errors = []
    assert import_p_to_o(path, chains, errors) == {}
    assert errors[0].reason == "p_to_o_mapping_invalid"


def test_p_lyso_is_explicit_switch(inputs):
    chains, polars, config = inputs
    bank = build_combination_bank(chains, replace(config, include_p_lyso=True))
    single = next(combo for combo in bank if combo.linkage_family == "P")
    assert permitted(single, polars["demo_PE"], replace(config, include_p_lyso=True))
    assert not permitted(single, polars["demo_PC"], replace(config, include_p_lyso=True))
