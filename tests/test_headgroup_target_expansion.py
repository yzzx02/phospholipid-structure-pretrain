"""Critical reaction, anchor, formula and identity checks for proposed targets."""

import pytest
from rdkit import Chem

from structure_pretrain.glycerol_core_collection.collection import legacy_core
from structure_pretrain.headgroup_target_expansion.chemistry import (
    amino_acid_from_pg_polar,
    aminoacylate,
    free_head_alcohols,
    local_edits,
    policy,
    representations,
    standard_core,
)


def core(polar):
    return standard_core(legacy_core(polar)["core_smiles_chain_sites"])[0]


def test_neutral_PC_source_protonation_merges_without_heavy_graph_change():
    a, notes = standard_core("C[N+](C)(C)CCOP(=O)(O)OCC(O[*:102])CO[*:101]")
    b, _ = standard_core("C[N+](C)(C)CCOP(=O)([O-])OCC(O[*:102])CO[*:101]")
    assert notes and Chem.MolToSmiles(a) == Chem.MolToSmiles(b)
    assert Chem.GetFormalCharge(a) == 0


def test_both_roots_roundtrip_multi_P_and_use_attachment_P():
    m = core("OP(=O)(O)OCCOP(=O)(O)[*:1]")
    r = representations(m)
    assert r["target_smiles"].startswith("P")
    assert r["glycerol_first_smiles"].startswith("C")
    h = Chem.MolFromSmiles(r["polar_P_first_smiles"])
    assert any(a.GetAtomMapNum() == 1 for a in h.GetAtomWithIdx(0).GetNeighbors())


def test_attached_roles_retain_distinct_PG_aminoacyl_positions():
    terminal = core("CC(N)C(=O)OCC(O)COP(=O)(O)[*:1]")
    central = core("CC(N)C(=O)OC(CO)COP(=O)(O)[*:1]")
    assert representations(terminal)["target_id"] != representations(central)["target_id"]


def test_alpha_carboxyl_donor_is_not_Asp_side_chain():
    acid = amino_acid_from_pg_polar("NC(CC(=O)O)C(=O)OCC(O)COP(=O)(O)[*:1]")
    assert Chem.MolToSmiles(acid) == "NC(CC(=O)O)C(=O)O"
    pg = core("OCC(O)COP(=O)(O)[*:1]")
    sites = list(free_head_alcohols(pg))
    assert len(sites) == 2
    products = [aminoacylate(pg, i, acid) for i in sites]
    assert len({representations(x)["target_id"] for x in products}) == 2
    assert all(policy(x)["core_formula"] == "C10H20NO11P" for x in products)


def test_direct_acyl_phosphate_is_valid_but_outside_default_proposal():
    m = core("CC(N)C(=O)OP(=O)(O)[*:1]")
    assert "reactive_acyl_phosphate_mixed_anhydride" in policy(m)["filter_reasons"]


def test_source_amine_methylation_and_carboxyl_ester_are_real_candidates():
    pe = core("NCCOP(=O)(O)[*:1]")
    outcomes = []
    for label, _, apply in local_edits(pe):
        if label == "N_methyl":
            outcomes.append(policy(apply())["core_formula"])
    assert "C6H16NO6P" in outcomes
    ps = core("NC(COP(=O)(O)[*:1])C(=O)O")
    products = [apply() for label, _, apply in local_edits(ps) if label == "O_ethyl"]
    assert any(
        Chem.MolFromSmarts("C(=O)OCC") and x.HasSubstructMatch(Chem.MolFromSmarts("C(=O)OCC"))
        for x in products
    )


def test_no_edit_rewrites_glycerol_or_adds_fatty_chain():
    m = core("OCC(O)COP(=O)(O)[*:1]")
    for label, site, apply in local_edits(m):
        try:
            product = apply()
        except (ValueError, RuntimeError):
            continue
        assert len([a for a in product.GetAtoms() if a.GetAtomMapNum() in (101, 102)]) == 2
        representations(product)


@pytest.mark.parametrize("text", ["OP(=O)(O)[*:1]", "OCC(O)COP(=O)(O)[*:1]"])
def test_frozen_new_mass_policy_is_500_without_sugar(text):
    assert policy(core(text))["mass_limit_Da"] == 500


def test_nonP_core_also_uses_500_and_has_glycerol_root():
    mol, _ = standard_core("OCC(O[*:102])COCCN")
    assert policy(mol)["P_count"] == 0 and policy(mol)["mass_limit_Da"] == 500
    assert representations(mol)["target_smiles"].startswith("C")
    assert any(label == "N_propargyl" for label, _, _ in local_edits(mol))


def test_nonP_sugar_keeps_600_and_ring_bonds_not_made_triple():
    mol, _ = standard_core("OCC(O[*:102])COC1OC(CO)C(O)C(O)C1O")
    assert policy(mol)["P_count"] == 0 and policy(mol)["mass_limit_Da"] == 600
    for label, sites, _ in local_edits(mol):
        if label == "C_C_to_triple":
            assert not mol.GetBondBetweenAtoms(*sites).IsInRing()


def test_short_acyl_second_glycerol_anchor_is_rejected_but_aminoacyl_is_retained():
    short_acyl = core("CC(=O)OCC(O)COP(=O)(O)[*:1]")
    aminoacyl = core("CC(N)C(=O)OCC(O)COP(=O)(O)[*:1]")
    reason = "extra_glycerol_lipid_anchor_including_short_acyl"
    assert reason in policy(short_acyl)["filter_reasons"]
    assert reason not in policy(aminoacyl)["filter_reasons"]


def test_fatty_N_acyl_PE_is_rejected_without_rejecting_aminoacyl_peptide():
    n_acetyl = core("CC(=O)NCCOP(=O)(O)[*:1]")
    gly_peptide = core("NCC(=O)NCCOP(=O)(O)[*:1]")
    assert "extra_fatty_N_acyl_PE_PS_head" in policy(n_acetyl)["filter_reasons"]
    assert "extra_fatty_N_acyl_PE_PS_head" not in policy(gly_peptide)["filter_reasons"]
