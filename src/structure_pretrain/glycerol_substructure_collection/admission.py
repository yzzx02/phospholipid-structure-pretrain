"""Explicit user admission overrides; preserve physical risk/mass annotations."""

MASS = "core_mass_above_500_600_policy"
ALLOWED_RISKS = frozenset(
    {
        "synthetic_protecting_group_suspected",
        "reactive_acyl_phosphate_mixed_anhydride",
    }
)


def readmission_group(row):
    flags = set(filter(None, row["new_flags"].split(";")))
    if row["v03_pass"]:
        return None
    if row["mass_shift_standby"] and flags == {MASS}:
        return "user_accepted_mass_only_exception"
    if flags and flags <= ALLOWED_RISKS:
        return "user_accepted_risk_annotations"
    return None
