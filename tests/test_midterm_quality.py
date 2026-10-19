from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_midterm_v03 import choose_stratified, relation_codes
from structure_pretrain.data.prepare_v03 import canonical_polar
from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import make_combination
from structure_pretrain.library_builder.chain_library import import_chains
from structure_pretrain.library_builder.polar_library import import_polars
from structure_pretrain.library_builder.schemas import ELEMENTS
from structure_pretrain.training.quality_v03 import polar_checks, score_candidate


@pytest.mark.parametrize(
    "text,eos,category",
    [
        ("[*:1]P(=O)(O)O", False, "no_eos"),
        ("[*:1]P(", True, "smiles_syntax"),
        ("OP(=O)(O)O", True, "attachment"),
        ("[*:2]P(=O)(O)O", True, "attachment"),
        ("[*:1]CP(=O)(O)O", True, "attachment"),
        ("[*:1]P(=O)([O-])O", True, "polar_or_builder_chemistry"),
        ("[*:1]P(=O)(O)O", True, None),
    ],
)
def test_separate_syntax_attachment_chemistry(text, eos, category):
    result = polar_checks(text, eos)
    assert result.get("category") == category
    assert result["polar_valid"] == (category is None)


def test_top10_can_include_multiple_transformation_types():
    codes = relation_codes(
        np.array([[1], [2]]),
        np.array([[2, 1, 2, 1], [3, 2, 3, 2]]),
        np.array([[[1, 2]], [[3, 4]]]),
        np.array([[[1, 2], [3, 4], [3, 4], [1, 2]], [[3, 4], [1, 2], [1, 2], [3, 4]]]),
    )
    np.testing.assert_array_equal(codes, [[1, 2, 3, 0], [1, 2, 3, 0]])


def test_diagnostic_sampling_reproducible_and_distinct_heads():
    eligible = np.arange(128, 640)
    strata = {int(i): f"family{i % 3}|length{i % 4}" for i in eligible}
    groups = np.arange(640) // 2
    selected = choose_stratified(eligible, strata, groups, count=128)
    repeated = choose_stratified(eligible, strata, groups, count=128)
    np.testing.assert_array_equal(selected, repeated)
    assert len(set(groups[selected])) == 128
    assert not set(selected) & set(range(128))
    assert len({strata[int(i)] for i in selected}) == 12


def test_complete_assembly_formula_and_isomer_failure_stages():
    example = Path(__file__).resolve().parents[1] / "examples/library_builder"
    errors = []
    chains = import_chains(example / "chains.tsv", errors)
    polar = import_polars(example / "polar_moieties.tsv", errors)[1]
    assert not errors
    text = polar.polar_fragment_smiles
    graph = canonical_polar(text)
    target = assemble(chains, make_combination(("A-14:0", "A-18:0"), chains), polar)
    formula = [target[f"neutral_formula_{e}"] for e in ELEMENTS]

    def score(tokens, expected_formula=formula):
        return score_candidate(
            text,
            True,
            tokens,
            chains,
            {graph: polar},
            expected_formula,
            graph,
            target["canonical_smiles"],
            {graph},
        )

    correct = score(["A-14:0", "A-18:0"])
    assert correct["category"] == "correct_structure"
    assert correct["morgan_similarity"] == 1
    assert correct["template_copy"]
    isomer = score(["A-16:0", "A-16:0"])
    assert isomer["formula_match"] and not isomer["full_exact"]
    assert isomer["category"] == "formula_matches_wrong_structure"
    assert score(["A-16:0", "NULL"])["category"] == "formula_mismatch"
    assert score(["NULL", "NULL"])["category"] == "chain_combination"
