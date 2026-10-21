"""Selection must distinguish extra lipid anchors from polar functional groups."""

import pytest

from structure_pretrain.reported_headgroups.single_anchor_review import analyze_single_anchor


@pytest.mark.parametrize(
    "smiles",
    [
        "C[N+](C)(C)CCOP(=O)([O-])[*:1]",
        "NC(COP(=O)(O)[*:1])C(=O)O",
        "NC(CCOP(=O)(O)[*:1])C(=O)O",
        "O=C(O)CNCCOP(=O)(O)[*:1]",
        "O=P(O)(O)OP(=O)(O)[*:1]",
        "O=P(O)(O)OC1C(O)C(OP(=O)(O)[*:1])C(O)C(OP(=O)(O)O)C1OP(=O)(O)O",
        "O=P(O)(OC1C(OC2OC(CO)C(O)C(O)C2O)C(O)C(O)C(O)C1OC1OC(CO)C(O)C(O)C1O)[*:1]",
    ],
)
def test_polar_acids_polyphosphates_and_glycans_not_extra_anchors(smiles):
    result = analyze_single_anchor(smiles)
    assert result["auto_status"] == "retain"
    assert result["highlight_extra_atoms"] == []


@pytest.mark.parametrize(
    "smiles,abbr",
    [
        ("CCCCCCCCCCCCCCCCCC(=O)NCCOP(=O)(O)[*:1]", "NAPE"),
        ("CCCCCCCCCCCCCCCC(=O)NC(COP(=O)(O)[*:1])C(=O)O", "PS-NAc"),
        ("CCCCCCCCCCCCCCCC(=O)OCC(O)COP(=O)(O)[*:1]", ""),
        ("CC(=O)OC(CO)COP(=O)(O)[*:1]", "BMP"),
        ("CCCCCCCCCCCCCCCCOCC(O)COP(=O)(O)[*:1]", ""),
    ],
)
def test_extra_amide_ester_ether_and_explicit_bmp_excluded(smiles, abbr):
    assert analyze_single_anchor(smiles, abbr)["auto_status"] == "exclude"


def test_sterol_anchor_is_excluded_even_if_builder_passes():
    smiles = "CC(C)CCCC(C)C1CCC2C3CC=C4CC(OC5OC(COP(=O)(O)[*:1])C(O)C(O)C5O)CCC4(C)C3CCC12C"
    result = analyze_single_anchor(smiles, builder_compatible=True)
    assert result["auto_status"] == "exclude"
    assert result["sterol_anchor_detected"]


@pytest.mark.parametrize(
    "smiles",
    [
        "NCCCCC(N)C(=O)OC(CO)COP(=O)(O)[*:1]",
        "CC(N)C(=O)OC(CO)COP(=O)(O)[*:1]",
        "CC(=O)OCC1OC(OP(=O)(O)[*:1])C(O)C(O)C1O",
    ],
)
def test_amino_acyl_and_short_acetyl_are_reviewed_not_called_fatty_chains(smiles):
    result = analyze_single_anchor(smiles)
    assert result["auto_status"] == "review"
    assert result["longest_hydrocarbon_path"] < 8
    assert result["highlight_extra_atoms"] == []


def test_builder_failure_and_source_conflict_stay_reviewable():
    assert (
        analyze_single_anchor("NCCOP(=O)(O)[*:1]", builder_compatible=False)["auto_status"]
        == "review"
    )
    assert (
        analyze_single_anchor("COC(=O)NCCOP(=O)(O)[*:1]", review_reason="name conflict")[
            "auto_status"
        ]
        == "review"
    )


def test_multiple_or_unrooted_attachments_are_not_silently_repaired():
    with pytest.raises(ValueError, match="single P"):
        analyze_single_anchor("O=P(O)([*:1])OCCOP(=O)(O)[*:2]")
