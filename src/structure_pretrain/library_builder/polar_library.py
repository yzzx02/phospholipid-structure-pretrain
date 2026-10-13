import csv
from dataclasses import replace
from pathlib import Path

from .canonicalize import nonisomeric_smiles
from .chain_library import boolean
from .schemas import PolarRule, Rejection
from .validation import validate_polar_fragment

FAMILIES = {"PE", "known_other", "pubchem_expanded", "phosphonate", "other"}


def import_polars(path: str | Path, rejections: list[Rejection]) -> list[PolarRule]:
    grouped, ids = {}, set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {
            "polar_id",
            "source",
            "source_id",
            "original_smiles",
            "polar_fragment_smiles",
            "attachment_atom_map",
            "polar_family",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing polar columns: {sorted(required - set(reader.fieldnames or []))}"
            )
        for row in reader:
            identifier = row["polar_id"]
            try:
                if not identifier or identifier in ids:
                    raise ValueError("Polar IDs must be nonempty and unique")
                ids.add(identifier)
                family = row["polar_family"]
                if family not in FAMILIES or not row["source"]:
                    raise ValueError("Known polar_family and source provenance are required")
                atom_map = int(row["attachment_atom_map"])
                molecule = validate_polar_fragment(row["polar_fragment_smiles"], atom_map)
                aa, oa = boolean(row.get("allow_aa"), True), boolean(row.get("allow_oa"), True)
                pa = boolean(row.get("allow_pa"), False)
                if pa and family != "PE":
                    raise ValueError(
                        "Independent P/A permission is restricted to externally designated PE"
                    )
                if not (aa or oa or pa):
                    raise ValueError("Polar has no permitted chain families")
                normalized = nonisomeric_smiles(molecule)
                # Ignore only attachment numbering when deduplicating identical polar chemistry.
                dummy = next(atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0)
                dummy.SetAtomMapNum(1)
                key = nonisomeric_smiles(molecule)
                rule = PolarRule(
                    identifier,
                    row["source"],
                    row["source_id"],
                    row["original_smiles"],
                    normalized,
                    atom_map,
                    family,
                    aa,
                    oa,
                    pa,
                    (identifier,),
                    (f"{row['source']}:{row['source_id']}",),
                )
                grouped.setdefault(key, []).append(rule)
            except (ValueError, TypeError) as exc:
                rejections.append(
                    Rejection(
                        "polar_import",
                        identifier,
                        getattr(exc, "reason", "polar_row_invalid"),
                        str(exc),
                    )
                )
    result = []
    for rules in grouped.values():
        rules.sort(key=lambda rule: rule.polar_id)
        first = rules[0]
        if any(
            (r.polar_family, r.allow_aa, r.allow_oa, r.allow_pa)
            != (first.polar_family, first.allow_aa, first.allow_oa, first.allow_pa)
            for r in rules
        ):
            for rule in rules:
                rejections.append(
                    Rejection(
                        "polar_import",
                        rule.polar_id,
                        "conflicting_polar_policy",
                        "Equivalent fragments have conflicting family or permission declarations",
                    )
                )
            continue
        result.append(
            replace(
                first,
                source_polar_ids=tuple(r.polar_id for r in rules),
                provenance=tuple(sorted({p for r in rules for p in r.provenance})),
            )
        )
    return sorted(result, key=lambda rule: rule.polar_id)
