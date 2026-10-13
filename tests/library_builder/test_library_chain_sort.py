from dataclasses import replace

import pytest

from structure_pretrain.library_builder.chain_combinations import make_combination
from structure_pretrain.library_builder.chain_library import chain_sort_key, validate_chain
from structure_pretrain.library_builder.schemas import LibraryValidationError


def test_chain_sort_and_unordered_aa(inputs):
    chains, _, _ = inputs
    assert chain_sort_key("A-16:0") < chain_sort_key("A-18:0") < chain_sort_key("A-18:1")
    left = make_combination(("A-18:1", "A-16:0"), chains)
    right = make_combination(("A-16:0", "A-18:1"), chains)
    assert left == right


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"carbon_count": 17}, "chain_token_mismatch"),
        ({"representative_double_bond_positions": (8,)}, "double_bond_positions_mismatch"),
        ({"representative_fragment_smiles": "[*:1]CCCC"}, "carbon_count_mismatch"),
    ],
)
def test_source_chain_consistency(inputs, change, reason):
    chains, _, _ = inputs
    with pytest.raises(LibraryValidationError) as error:
        validate_chain(replace(chains["A-18:1"], **change))
    assert error.value.reason == reason


def test_p_requires_external_db_convention(inputs):
    chains, _, _ = inputs
    with pytest.raises(LibraryValidationError, match="explicit external"):
        validate_chain(replace(chains["P-16:0"], p_db_count_includes_vinyl=None))
