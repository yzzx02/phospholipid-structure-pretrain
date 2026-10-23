"""Chemical boundary and identity tests, independent of every frozen training input."""

import pytest
from rdkit.Chem import rdMolDescriptors

from structure_pretrain.glycerol_core_collection.chemistry import (
    canonical,
    capped_core,
    connectivity_key,
    core_definition,
    descriptors,
    extract_glycerol_core,
    glycosyl_rule_candidates,
    parse,
    root_phosphate,
)
from structure_pretrain.glycerol_core_collection.collection import legacy_core

PC = "CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC"


def test_actual_source_chain_cut_preserves_stereo_formula_and_slots():
    r = extract_glycerol_core(PC)
    assert r["parent_roundtrip_stereo"] and r["parent_roundtrip_formula_including_H"]
    assert r["acyl_chain_count"] == 2
    assert r["core_display_smiles"].startswith("O=P")
    assert descriptors(r["capped_core_smiles"])["core_formula"] == "C8H20NO6P"
    assert {t["slot"] for t in r["original_chain_fragments"]} == {101, 102}


def test_one_id_ignores_stereo_but_retains_position():
    assert connectivity_key("OC[C@H](O)COCCN") == connectivity_key("OC[C@@H](O)COCCN")
    assert connectivity_key("OCC(O)COCCN") != connectivity_key("OCC(OCCN)CO")


def test_multi_P_roots_at_marker_associated_P_and_preserves_graph():
    text = "OP(=O)(O)OCCOP(=O)(O)[*:1]"
    display, _ = root_phosphate(text)
    assert display.startswith("O=P")
    m = parse(display)
    first_P = m.GetAtomWithIdx(0).GetNeighbors()[0]
    assert any(a.GetAtomMapNum() == 1 for a in first_P.GetNeighbors())
    assert canonical(m) == canonical(parse(text))


def test_thiophosphate_never_rewritten_as_oxo():
    text = "S=P(O)(OCCN)[*:1]"
    display, status = root_phosphate(text)
    assert "graph_preserved" in status and canonical(parse(display)) == canonical(parse(text))


def test_sugar_and_inositol_have_600_other_400():
    assert descriptors("OC1C(O)C(O)C(O)C(O)C1O")["mass_limit_Da"] == 600
    assert descriptors("OCC1OC(O)C(O)C(O)C1O")["mass_limit_Da"] == 600
    assert descriptors("NCCO")["mass_limit_Da"] == 400


def test_extra_head_fatty_chain_detected_aminoacyl_not_misclassified():
    assert (
        "extra_head_long_alkyl_or_acyl_chain"
        in descriptors("CCCCCCCCCC(=O)NCCO")["structural_filter_reasons"]
    )
    assert (
        "extra_head_long_alkyl_or_acyl_chain"
        not in descriptors("NCCCCC(N)C(=O)OCC(O)CO")["structural_filter_reasons"]
    )


def test_aminoacyl_PG_is_not_a_second_fatty_anchor():
    text = "CCCCCCCC(=O)OCC(OC(=O)CCCCCCC)COP(=O)(O)OCC(OC(=O)C(N)C)CO"
    r = extract_glycerol_core(text)
    assert r["chain_count"] == 2
    assert not descriptors(r["capped_core_smiles"])["structural_filter_reasons"]


def test_legacy_expansion_keeps_explicit_attachment_and_has_no_old_label_guard():
    r = legacy_core("CC(N)C(=O)OCC(O)COP(=O)(O)[*:1]")
    assert r["core_display_smiles"].startswith("O=P")
    assert "N" in r["capped_core_smiles"]
    assert (
        len([a for a in parse(r["core_smiles_chain_sites"]).GetAtoms() if a.GetAtomicNum() == 0])
        == 2
    )


def test_definition_rejects_extra_anchor_and_N_socket():
    with pytest.raises(ValueError):
        core_definition("OCC(O[*:101])COP(=O)(O)OCC(N[*:102])CO")
    with pytest.raises(ValueError):
        core_definition("OCC(O[*:101])COP(=O)(O)OCC(O[*:102])CO")


def test_two_actual_lipid_anchors_rejected_even_short_source_acyls():
    with pytest.raises(ValueError, match="multiple_lipid_glycerol_anchors"):
        extract_glycerol_core("CC(=O)OCC(OC(C)=O)COP(=O)(O)OCC(OC(C)=O)COC(C)=O")


def test_halogen_and_protecting_group_hold_are_explicit():
    assert "halogen_hold_for_natural_evidence" in descriptors("OCCF")["structural_filter_reasons"]
    assert (
        "synthetic_protecting_group_suspected"
        in descriptors("CC(C)(C)OC(=O)NCCO")["structural_filter_reasons"]
    )


def test_glycosidic_rule_conserves_formula_and_never_claims_observation():
    r = glycosyl_rule_candidates("OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O")
    assert len(r) == 1 and not r[0]["full_lipid_observed"]
    assert rdMolDescriptors.CalcMolFormula(parse(r[0]["capped_core_smiles"])) == "C9H18O8"
    assert not r[0]["parent_roundtrip_formula_including_H"]
    assert (
        canonical(capped_core(parse(r[0]["core_smiles_chain_sites"]))) == r[0]["capped_core_smiles"]
    )


def test_linear_donor_is_not_guessed_into_ring_or_new_linkage():
    assert glycosyl_rule_candidates("OCC(O)C(O)C(O)C(O)C=O") == []
