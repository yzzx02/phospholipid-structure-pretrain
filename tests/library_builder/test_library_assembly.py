from dataclasses import replace

import pytest
from rdkit import Chem

from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import make_combination
from structure_pretrain.library_builder.chain_library import validate_chain
from structure_pretrain.library_builder.schemas import ChainRule, LibraryValidationError
from structure_pretrain.library_builder.validation import validate_polar_fragment


def test_aa_swapping_gives_identical_full_structure(inputs):
    chains, polars, _ = inputs
    combo = make_combination(("A-16:0", "A-18:1"), chains)
    left = assemble(chains, combo, polars["demo_PC"])
    right = assemble(
        chains, replace(combo, chain_tokens=tuple(reversed(combo.chain_tokens))), polars["demo_PC"]
    )
    assert left["canonical_smiles"] == right["canonical_smiles"]
    assert left["inchikey"] == right["inchikey"]
    assert "*" not in left["canonical_smiles"] and "@" not in left["canonical_smiles"]


def test_phosphonate_has_direct_p_c_bond(inputs):
    chains, polars, _ = inputs
    row = assemble(
        chains, make_combination(("A-16:0", "A-18:1"), chains), polars["demo_phosphonate"]
    )
    mol = Chem.MolFromSmiles(row["canonical_smiles"])
    assert mol.HasSubstructMatch(Chem.MolFromSmarts("[P]-[C]"))
    assert row["neutral_formula_P"] == 1


def test_stereo_removed_but_double_bond_positions_retained(inputs):
    chains, polars, _ = inputs
    rules = [
        ChainRule("O-8:1", "O", 8, 1, text, (position,), "explicit_test_fixture")
        for text, position in [
            ("[*:1]CCC/C=C/CCC", 4),
            ("[*:1]CCC/C=C\\CCC", 4),
            ("[*:1]CC/C=C/CCCC", 3),
        ]
    ]
    assembled = []
    for rule in rules:
        library = dict(chains, **{rule.chain_token: validate_chain(rule)})
        assembled.append(
            assemble(
                library, make_combination((rule.chain_token, "A-16:0"), library), polars["demo_PC"]
            )
        )
    assert assembled[0]["canonical_smiles"] == assembled[1]["canonical_smiles"]
    assert assembled[0]["canonical_smiles"] != assembled[2]["canonical_smiles"]
    assert assembled[0]["neutral_formula"] == assembled[2]["neutral_formula"]


@pytest.mark.parametrize("fragment", ["P(=O)(O)O", "[*:1]P(=O)(O)O[*:2]", "[*:1]OP(=O)(O)O"])
def test_attachment_contract(fragment):
    with pytest.raises(LibraryValidationError):
        validate_polar_fragment(fragment, 1)


def test_reject_non_neutral_complete_assembly(inputs):
    chains, polars, _ = inputs
    charged = replace(polars["demo_PC"], polar_fragment_smiles="[*:1]P(=O)(O)OCC[N+](C)(C)C")
    with pytest.raises(LibraryValidationError) as error:
        assemble(chains, make_combination(("A-16:0", "A-18:1"), chains), charged)
    assert error.value.reason == "nonneutral_structure"
