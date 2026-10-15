from pathlib import Path

import pytest
import yaml
from rdkit import Chem

from structure_pretrain.pubchem_polar.chemistry import (
    Reject,
    process_smiles,
    read_controls,
    validate_controls,
)
from structure_pretrain.pubchem_polar.pipeline import merge_candidate

ROOT = Path(__file__).resolve().parents[2]


def keys(smiles):
    return {r["polar_fragment_smiles"] for r in process_smiles(smiles)["candidates"]}


def test_all_local_controls_and_configured_queries():
    config = yaml.safe_load((ROOT / "configs/pubchem_polar_queries.yaml").read_text())
    result, _ = validate_controls(config, read_controls(ROOT / "configs/known_head_controls.tsv"))
    assert len(result) == 14
    assert all(r["passed"] for r in result.values())


@pytest.mark.parametrize(
    "smiles,family",
    [
        ("OP(=O)(O)OCCN", "phosphate"),
        ("OP(=O)(O)CCN", "phosphonate"),
        ("OP(=O)(O)OP(=O)(O)O", "polyphosphate"),
    ],
)
def test_attachment_directly_to_p(smiles, family):
    result = process_smiles(smiles)
    for candidate in result["candidates"]:
        mol = Chem.MolFromSmiles(candidate["polar_fragment_smiles"])
        dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        assert len(dummies) == 1 and dummies[0].GetAtomMapNum() == 1
        assert dummies[0].GetDegree() == 1
        assert dummies[0].GetNeighbors()[0].GetAtomicNum() == 15
        assert candidate["polar_family"] == family


def test_no_poc_or_pc_cleavage():
    for source, pattern in [("OP(=O)(O)OCCN", "POCCN"), ("OP(=O)(O)CCN", "PCCN")]:
        for smi in keys(source):
            assert Chem.MolFromSmiles(smi).HasSubstructMatch(Chem.MolFromSmarts(pattern))
    with pytest.raises(Reject, match="no_acidic_attachment"):
        process_smiles("COP(=O)(OC)OC")


def test_equivalent_sites_deduplicate():
    result = process_smiles("OP(=O)(O)O")
    assert len(result["candidates"]) == 3
    assert len(keys("OP(=O)(O)O")) == 1


def test_pc_charge_forms_and_attachment_sites_converge():
    sources = [
        "OP(=O)([O-])OCC[N+](C)(C)C",
        "OP(=O)(O)OCC[N+](C)(C)C",
        "[O-]P(=O)([O-])OCC[N+](C)(C)C",
    ]
    assert len(set.union(*(keys(s) for s in sources))) == 1
    for source in sources:
        for candidate in process_smiles(source)["candidates"]:
            mol = Chem.MolFromSmiles(candidate["polar_fragment_smiles"])
            assert Chem.GetFormalCharge(mol) == 0
            assert any(a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1 for a in mol.GetAtoms())
            assert not candidate["charge_ambiguous"]


def test_reversible_pe_zwitterion_converges():
    assert keys("OP(=O)(O)OCCN") == keys("OP(=O)([O-])OCC[NH3+]")


def test_simple_salt_removed_but_organic_component_preserved_as_rejection():
    result = process_smiles("[Na+].[O-]P(=O)(O)OCCN")
    assert result["removed_counterions"] == ["[Na+]"]
    assert keys("[Na+].[O-]P(=O)(O)OCCN") == keys("OP(=O)(O)OCCN")
    with pytest.raises(Reject, match="CCO"):
        process_smiles("OP(=O)(O)O.CCO")


def test_multiple_disconnected_p_rejected():
    with pytest.raises(Reject, match="multiple_p_components"):
        process_smiles("OP(=O)(O)O.OP(=O)(O)O")


def test_long_alkyl_flagged():
    result = process_smiles("CCCCCCCCCCCCOP(=O)(O)O")
    assert all(
        c["status"] == "FILTER_REVIEW" and c["max_aliphatic_run"] == 12
        for c in result["candidates"]
    )


@pytest.mark.parametrize(
    "smiles", ["OP(=O)(O)OC1C(O)C(O)C(O)C(O)C1O", "OP(=O)(O)OCC(N)C(=O)O", "OP(=O)(O)OCc1ccncc1"]
)
def test_sugar_amino_acid_heterocycle_allowed(smiles):
    assert all(c["status"] == "APPROVABLE" for c in process_smiles(smiles)["candidates"])


def test_stereo_does_not_duplicate():
    assert keys("OP(=O)(O)OC[C@H](N)C(=O)O") == keys("OP(=O)(O)OC[C@@H](N)C(=O)O")
    assert keys("OP(=O)(O)OC/C=C/C") == keys("OP(=O)(O)OC/C=C\\C")


def test_multiple_inequivalent_attachments_marked():
    candidates = process_smiles("OP(=O)(O)OCCOP(=O)(O)OC")["candidates"]
    assert len({c["polar_fragment_smiles"] for c in candidates}) == 2
    assert all(c["status"] == "MULTIPLE_ATTACHMENT_AMBIGUITY" for c in candidates)


@pytest.mark.parametrize(
    "smiles,reason",
    [
        ("OP(=O)(O)C[13CH3]", "isotope_labels"),
        ("OP(=O)(O)C[CH2]", "radicals"),
        ("OP(=O)(O)[SiH3]", "unsupported_elements"),
        ("OP(=O)(C)C", "unsupported_phosphorus_chemistry"),
        ("OP(=S)(O)O", "unsupported_phosphorus_chemistry"),
        ("OP(=O)(O)N", "unsupported_phosphorus_chemistry"),
    ],
)
def test_hard_rejections(smiles, reason):
    with pytest.raises(Reject) as caught:
        process_smiles(smiles)
    assert caught.value.reason == reason


def test_unbalanced_fixed_charge_review_only():
    candidates = process_smiles("COP(=O)(O)OCC[N+](C)(C)C")["candidates"]
    assert all(c["status"] == "CHARGE_AMBIGUOUS" for c in candidates)


def test_acid_charge_location_does_not_hide_ambiguity():
    sources = ["OP(=O)([O-])OCC[N+](C)(C)CC(=O)O", "OP(=O)(O)OCC[N+](C)(C)CC(=O)[O-]"]
    assert keys(sources[0]) == keys(sources[1])
    assert all(c["charge_ambiguous"] for s in sources for c in process_smiles(s)["candidates"])


def test_cid_provenance_merge_and_pe_permission():
    c = process_smiles("OP(=O)(O)OCCN")["candidates"][0]
    fragments = {}
    controls = {"PE": [c["polar_fragment_smiles"]]}
    merge_candidate(fragments, c, 100, ["a"], controls)
    merge_candidate(fragments, c, 200, ["b"], controls)
    merge_candidate(fragments, c, 100, ["b"], controls)
    row = fragments[c["polar_fragment_smiles"]]
    assert row["source_cids"] == {100, 200}
    assert row["matched_query_ids"] == {"a", "b"}
    assert row["allow_pa"]
    other = process_smiles("OP(=O)(O)OCCNC")["candidates"][0]
    merge_candidate(fragments, other, 300, ["a"], controls)
    assert not fragments[other["polar_fragment_smiles"]]["allow_pa"]
