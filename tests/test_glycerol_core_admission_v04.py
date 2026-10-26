from structure_pretrain.glycerol_substructure_collection.admission import (
    ALLOWED_RISKS,
    MASS,
    readmission_group,
)


def row(flags, *, shift=False, kept=False):
    return {"new_flags": ";".join(sorted(flags)), "mass_shift_standby": shift, "v03_pass": kept}


def test_only_explicit_mass_shift_group_is_reinstated():
    assert readmission_group(row({MASS}, shift=True)) == "user_accepted_mass_only_exception"
    assert readmission_group(row({MASS})) is None


def test_mass_plus_other_flags_is_still_standby():
    for flag in ALLOWED_RISKS | {"extra_head_long_alkyl_or_acyl_chain"}:
        assert readmission_group(row({MASS, flag}, shift=True)) is None


def test_each_risk_and_their_overlap_are_accepted_once():
    for flags in [set(ALLOWED_RISKS), *[{x} for x in ALLOWED_RISKS]]:
        assert readmission_group(row(flags)) == "user_accepted_risk_annotations"


def test_accepted_risk_does_not_override_extra_lipid_anchors():
    for hard in (
        "extra_head_long_alkyl_or_acyl_chain",
        "extra_fatty_N_acyl_PE_PS_head",
        "extra_glycerol_lipid_anchor_including_short_acyl",
    ):
        assert readmission_group(row(set(ALLOWED_RISKS) | {hard})) is None


def test_existing_kept_graphs_are_not_duplicated_as_readmissions():
    assert readmission_group(row(set(), kept=True)) is None
