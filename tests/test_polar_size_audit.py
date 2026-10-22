from collections import Counter

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from structure_pretrain.reported_headgroups.size_audit import (
    PROFILES,
    chemistry_reasons,
    describe_polar,
    profile_reasons,
)


@pytest.mark.parametrize(
    "smiles",
    [
        "NCCOP(=O)(O)[*:1]",
        "C[N+](C)(C)CCOP(=O)([O-])[*:1]",
        "O=P(O)(O)[*:1]",
        "O=P(O)(O)OP(=O)(O)[*:1]",
    ],
)
def test_open_formula_matches_explicit_hydrogens_and_capping_mass(smiles):
    row = describe_polar(smiles)
    mol = Chem.MolFromSmiles(smiles)
    counter = Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms() if a.GetAtomicNum())
    for element in ("C", "H", "N", "O", "P"):
        assert row[element] == counter[element]
    rw = Chem.RWMol(mol)
    marker = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    rw.ReplaceAtom(marker, Chem.Atom(8))
    capped = rw.GetMol()
    Chem.SanitizeMol(capped)
    assert row["OH_capped_average_mass_Da"] == pytest.approx(Descriptors.MolWt(capped))
    assert row["heavy_atoms"] == mol.GetNumHeavyAtoms()


def test_carbon_free_phosphate_and_zwitterion_are_not_weird_formulas():
    pa = describe_polar("O=P(O)(O)[*:1]")
    assert pa["C"] == 0 and pa["H_C_ratio"] is None
    assert profile_reasons(pa, PROFILES["compact_300"]) == []
    pc = describe_polar("C[N+](C)(C)CCOP(=O)([O-])[*:1]")
    assert pc["charged_atom_count"] == 2 and pc["formal_charge"] == 0
    assert not chemistry_reasons(pc)


def test_low_mass_halogen_and_long_chain_need_more_than_mass_cutoff():
    fluorinated = describe_polar("FCCCOP(=O)(O)[*:1]")
    assert fluorinated["average_mass_Da"] < 300
    assert "outside_CHNOPS" in chemistry_reasons(fluorinated)
    extra_chain = describe_polar("CCCCCCCCCCCCOP(=O)(O)[*:1]")
    assert extra_chain["average_mass_Da"] < 300
    assert "extra_lipid_anchor" in chemistry_reasons(extra_chain)


def test_root_distance_uses_marked_p_and_allows_multiple_phosphorus():
    row = describe_polar("O=P(O)(O)OP(=O)(O)[*:1]")
    assert row["P"] == 2 and row["P_to_farthest_atom_bonds"] == 3
    with pytest.raises(ValueError, match="P-\\["):
        describe_polar("O=P(O)([*:1])OCCOP(=O)(O)[*:2]")
