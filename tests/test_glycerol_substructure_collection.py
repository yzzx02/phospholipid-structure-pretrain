from rdkit import Chem

from structure_pretrain.glycerol_core_collection.chemistry import extra_head_chain
from structure_pretrain.glycerol_substructure_collection.chemistry import (
    acyclic_long_chain,
    extract_observed_cores,
    policy,
    representations,
    verify_assembly,
)


def one(text):
    rows, _ = extract_observed_cores(text)
    assert len(rows) == 1
    assert rows[0]["source_roundtrip_H_formula_pass"]
    return rows[0]["mol"]


def test_no_fatty_chain_record_is_required():
    mol = one("OCC(O)COC")
    assert not policy(mol, new_pubchem_branch=True)["filter_reasons"]
    assert verify_assembly(mol)["full_assembly_H_formula_pass"]


def test_central_head_is_preserved_but_excluded_from_new_branch():
    mol = one("OCC(OC)CO")
    assert representations(mol)["head_attachment_position"] == "central"
    assert "sn2_head_attachment_excluded" in policy(mol, new_pubchem_branch=True)["filter_reasons"]


def test_terminal_ends_are_the_same_without_stereochemistry():
    a = one("COC[C@H](O)CO")
    b = one("COC[C@@H](O)CO")
    assert representations(a)["target_id"] == representations(b)["target_id"]


def test_P_source_does_not_enter_new_nonP_branch():
    mol = one("OCC(O)COP(=O)(O)O")
    assert (
        "P_containing_source_outside_new_nonP_branch"
        in policy(mol, new_pubchem_branch=True)["filter_reasons"]
    )


def test_allowed_halogen_is_not_held_for_natural_evidence():
    mol = one("OCC(O)COCCCl")
    assert not policy(mol, new_pubchem_branch=True)["filter_reasons"]


def test_ten_elements_is_not_a_ten_halogen_atoms_cap():
    mol = one("OCC(O)COC1(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C1(F)F")
    result = policy(mol, new_pubchem_branch=True)
    assert result["halogen_count"] == 11
    assert not result["filter_reasons"]


def test_amino_acid_ester_is_not_erased_as_a_fatty_chain():
    mol = one("NCC(=O)OCC(O)CO")
    assert any(a.GetAtomicNum() == 7 for a in mol.GetAtoms())
    assert not policy(mol, new_pubchem_branch=True)["filter_reasons"]


def test_polyfunctional_oxyacid_ester_remains_a_head():
    mol = one("OCC(O)COC(=O)C(O)CO")
    assert not policy(mol, new_pubchem_branch=True)["filter_reasons"]


def test_observed_fatty_chains_are_cut_and_source_roundtrip_passes():
    mol = one("CCCCCCCC(=O)OCC(OC(=O)CCCCCCC)COC")
    plain = one("OCC(O)COC")
    assert representations(mol)["target_id"] == representations(plain)["target_id"]


def test_a_monoacylglycerol_without_a_nonfatty_head_is_not_a_new_head():
    rows, _ = extract_observed_cores("CCCCCCCC(=O)OCC(O)CO")
    assert not rows


def test_carbocyclic_ring_is_not_an_acyclic_fatty_chain():
    source = Chem.MolFromSmiles("OC1CCCCCCC1")
    assert extra_head_chain(source)
    assert not acyclic_long_chain(source)
    mol = one("OCC(O)COC1CCCCCCC1")
    assert not policy(mol, new_pubchem_branch=True)["filter_reasons"]


def test_multiple_nonchain_substituents_are_not_silently_cut():
    rows, statuses = extract_observed_cores("COCC(OC)CO")
    assert not rows
    assert statuses["multiple_nonchain_glycerol_substituents_pending"]


def test_explicit_substructure_mode_can_cut_other_glycerol_substituents():
    rows, _ = extract_observed_cores("COCC(OC)CO", allow_other_site_substructure_cuts=True)
    admitted = [r for r in rows if not policy(r["mol"], new_pubchem_branch=True)["filter_reasons"]]
    assert len(admitted) == 1
    assert admitted[0]["source_nonfatty_substituent_cut_count"] == 1
    assert admitted[0]["observed_fatty_chain_count"] == 0
    assert admitted[0]["source_roundtrip_H_formula_pass"]
    assert (
        representations(admitted[0]["mol"])["target_id"]
        == representations(one("COCC(O)CO"))["target_id"]
    )


def test_two_observed_terminal_head_branches_remain_separate_candidates():
    rows, _ = extract_observed_cores("COCC(OC)COCCN", allow_other_site_substructure_cuts=True)
    admitted = [r for r in rows if not policy(r["mol"], new_pubchem_branch=True)["filter_reasons"]]
    assert len(admitted) == 2
    assert all(r["source_nonfatty_substituent_cut_count"] == 2 for r in admitted)
    assert all(r["source_roundtrip_H_formula_pass"] for r in admitted)


def test_unsupported_element_in_a_cut_group_does_not_reject_the_retained_head():
    rows, _ = extract_observed_cores(
        "COCC(O[Si](C)(C)C)CO", allow_other_site_substructure_cuts=True
    )
    admitted = [r for r in rows if not policy(r["mol"], new_pubchem_branch=True)["filter_reasons"]]
    assert len(admitted) == 1
    assert all(a.GetAtomicNum() != 14 for a in admitted[0]["mol"].GetAtoms())
    assert admitted[0]["source_roundtrip_H_formula_pass"]


def test_counterion_does_not_hide_a_valid_glycerol_head_component():
    rows, _ = extract_observed_cores(
        "OCC(O)COCC[NH3+].[Cl-]", allow_other_site_substructure_cuts=True
    )
    admitted = [r for r in rows if not policy(r["mol"], new_pubchem_branch=True)["filter_reasons"]]
    assert len(admitted) == 1
    assert admitted[0]["source_parent_had_multiple_components"]
    assert admitted[0]["source_roundtrip_H_formula_pass"]
