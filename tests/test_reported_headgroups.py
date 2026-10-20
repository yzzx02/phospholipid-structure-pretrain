import pytest
from rdkit import Chem

from structure_pretrain.reported_headgroups.chemistry import (
    builder_trial,
    extract_full,
    lsg_fragment,
    stable_id,
)


def test_source_PE_roundtrip_and_independent_H_formula():
    source = "CCCC(=O)OCC(OC(=O)CCC)COP(=O)(O)OCCN"
    (result,) = extract_full(source)
    assert result["polar_smiles"] == "NCCOP(=O)(O)[*:1]"
    assert result["roundtrip_graph"] and result["roundtrip_formula"]
    assert builder_trial(result["polar_smiles"])["independent_element_H_balance"]


def test_terminal_attachment_P_is_not_first_P():
    opened = "COP(=O)(O)OP(=O)(O)"
    polar = lsg_fragment(opened)["polar_smiles"]
    good = Chem.MolToSmiles(Chem.MolFromSmiles("COP(=O)(O)OP(=O)(O)[*:1]"))
    wrong = Chem.MolToSmiles(Chem.MolFromSmiles("COP(=O)([*:1])OP(=O)(O)O"))
    assert polar == good and polar != wrong
    source = opened + "OCC(OC(=O)CCC)COC(=O)CCC"
    assert extract_full(source)[0]["polar_smiles"] == polar


def test_chain_changes_do_not_create_new_headgroups():
    a = extract_full("CCCC(=O)OCC(OC(=O)CCC)COP(=O)(O)OCCN")[0]["polar_smiles"]
    b = extract_full("CCCCCC(=O)OCC(OC(=O)CCC)COP(=O)(O)OCCN")[0]["polar_smiles"]
    assert stable_id(a) == stable_id(b)


def test_cardiolipin_keeps_other_phosphatidyl_graph_and_builder_rejection():
    source = "CCCC(=O)OCC(OC(=O)CCC)COP(=O)(O)OCC(O)COP(=O)(O)OCC(OC(=O)CCC)COC(=O)CCC"
    (result,) = extract_full(source)
    assert "C(=O)" in result["polar_smiles"] or "(=O)O" in result["polar_smiles"]
    assert not result["builder_compatible"]
    assert result["incompatibility_reason"] == "intact_lipid_as_polar"


def test_named_PubChem_entry_without_lipid_scaffold_is_not_guessed():
    assert extract_full("C(CCN)CC(C(=O)OP(=O)(O)OCC(CO)O)N") == []
    with pytest.raises(ValueError, match="isotope_label_requires_explicit_review"):
        extract_full("CCCC(=O)OCC(OC(=O)CCC)COP(=O)(O)OCC[N+]([2H])(C)C")


def test_phosphoserine_head_is_not_misidentified_as_sphingoid_backbone():
    results = extract_full("CCCC(=O)OCC(OC(=O)CCC)COP(=O)(O)OCC(N)C(=O)O")
    assert len(results) == 1
    assert results[0]["polar_smiles"] == "NC(COP(=O)(O)[*:1])C(=O)O"


def test_actual_sphingomyelin_has_phosphocholine_head():
    results = extract_full("CCCCCCCCCCCCCCCC(=O)NC(COP(=O)([O-])OCC[N+](C)(C)C)C(O)CCCCCCCCCCCCCCC")
    assert len(results) == 1
    assert results[0]["polar_smiles"] == "C[N+](C)(C)CCOP(=O)([O-])[*:1]"
    assert results[0]["boundary_method"] == "observed_sphingoid_O_P_cut"


def test_isomeric_heads_are_not_deduplicated_by_formula():
    a = lsg_fragment("NCCOP(O)(=O)")["polar_smiles"]
    b = lsg_fragment("NC(C)OP(O)(=O)")["polar_smiles"]
    assert stable_id(a) != stable_id(b)
