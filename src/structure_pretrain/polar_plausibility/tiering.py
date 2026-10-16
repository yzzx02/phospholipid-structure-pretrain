from structure_pretrain.library_builder.canonicalize import nonisomeric_smiles
from structure_pretrain.library_builder.schemas import LibraryValidationError
from structure_pretrain.library_builder.validation import validate_fragment, validate_polar_fragment

from .descriptors import describe
from .reactive_rules import reactive_flags


def threshold_failures(descriptors, limits):
    failures = []
    for rule, threshold in limits.items():
        direction, name = rule.split("_", 1)
        if name == "aliphatic_carbon_run":
            name = "max_aliphatic_carbon_run"
        value = descriptors[name]
        if (direction == "max" and value > threshold) or (direction == "min" and value < threshold):
            failures.append(rule)
        if direction not in {"max", "min"}:
            raise ValueError(f"Unknown threshold direction: {rule}")
    return sorted(failures)


def assign_tier(descriptors, flags, config):
    if flags:
        return "REJECT", list(flags)
    core_failures = threshold_failures(descriptors, config["core"])
    if not core_failures:
        return "CORE", ["WITHIN_CORE_THRESHOLDS"]
    extended_failures = threshold_failures(descriptors, config["extended"])
    if not extended_failures:
        return "EXTENDED", [f"CORE_EXCEEDED:{name}" for name in core_failures]
    return "HOLD", [f"EXTENDED_EXCEEDED:{name}" for name in extended_failures]


def evaluate(row, config, rules):
    if row["status"] != "APPROVABLE":
        raise ValueError("Second-layer input must be status == APPROVABLE")
    result = dict(row)
    try:
        molecule, _, neighbor = validate_fragment(row["polar_fragment_smiles"], 1)
        if molecule.GetAtomWithIdx(neighbor).GetAtomicNum() != 15:
            raise ValueError("Polar boundary must attach to phosphorus")
        smiles = nonisomeric_smiles(molecule)
        round_trip, _, _ = validate_fragment(smiles, 1)
        if nonisomeric_smiles(round_trip) != smiles:
            raise ValueError("Non-idempotent SMILES round trip")
        result.update(describe(round_trip))
        result["reactive_flags"] = reactive_flags(round_trip, rules)
        result["tier"], result["tier_reasons"] = assign_tier(
            result, result["reactive_flags"], config
        )
        result["builder_boundary_hold"] = False
        if result["tier"] in {"CORE", "EXTENDED"}:
            try:
                validate_polar_fragment(smiles, 1)
            except LibraryValidationError as exc:
                if exc.reason != "intact_lipid_as_polar":
                    raise
                # Preserve graph-valid chemistry the unchanged mono-phosphatidyl builder
                # cannot accept. This is a compatibility HOLD, never a reactive rejection.
                result["tier"] = "HOLD"
                result["tier_reasons"] = ["BUILDER_BOUNDARY_REVIEW:intact_lipid_as_polar"]
                result["builder_boundary_hold"] = True
        result["evaluation_error"] = ""
    except Exception as exc:
        result.update(
            tier="REJECT",
            tier_reasons=["REJECT_ROUND_TRIP_OR_RULE_EXCEPTION"],
            reactive_flags=[],
            builder_boundary_hold=False,
            evaluation_error=f"{type(exc).__name__}: {exc}",
        )
    return result
