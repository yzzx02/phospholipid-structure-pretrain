"""Describe the frozen Polar library and compare proposed inference scopes; never filter it."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import rdBase

from structure_pretrain.reported_headgroups.size_audit import (
    PROFILES,
    chemistry_reasons,
    describe_polar,
    profile_reasons,
)

METRICS = (
    "average_mass_Da",
    "monoisotopic_mass_Da",
    "C",
    "H",
    "N",
    "O",
    "P",
    "S",
    "heavy_atoms",
    "rotatable_bonds",
    "P_to_farthest_atom_bonds",
    "smiles_tokens",
    "hydrocarbon_path",
    "ring_count",
    "aromatic_ring_count",
    "H_C_ratio",
    "O_C_ratio",
)
QUANTILES = (0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def quantiles(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return dict(zip(map(str, QUANTILES), map(float, np.quantile(values, QUANTILES))))


def references(library, review):
    result = {}
    for group, file in (
        ("human_approved_52", library / "approved_v01/approved_headgroups_v02.csv"),
        ("new_unreviewed_87", review / "new_review_candidates.csv"),
    ):
        with file.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        records = []
        for row in rows:
            records.append(
                {
                    "headgroup_id": row["headgroup_id"],
                    "name": row.get("preferred_review_name")
                    or row.get("display_name")
                    or row["full_name"],
                    "abbreviation": row["abbreviation"],
                    "polar_smiles": row["polar_smiles"],
                    **describe_polar(row["polar_smiles"]),
                }
            )
        result[group] = records
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/structure_library_v0.3/polar_library.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/polar_size_scope_audit_20261022")
    )
    parser.add_argument(
        "--reference-library", type=Path, default=Path("outputs/reported_headgroups_v02_20261021")
    )
    parser.add_argument(
        "--review", type=Path, default=Path("outputs/reported_headgroups_v02_review_20261021")
    )
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "summary.json").exists():
        raise ValueError("Choose a new output directory; existing analysis is preserved")
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    before = digest(args.input)
    assert before == "2948f729ae7cda9103d83ae220fe3673d2100d90e1fcb2290ad51fe7f12a8815"
    source = pq.read_table(
        args.input,
        columns=[
            "polar_id",
            "polar_fragment_smiles",
            "chemistry_family",
            "structure_origin",
            "polar_plausibility_tier",
        ],
    ).to_pylist()
    assert len(source) == 195306
    rows = []
    for i, row in enumerate(source, 1):
        values = describe_polar(row["polar_fragment_smiles"])
        values.update(row)
        for name, limits in PROFILES.items():
            values[name] = not profile_reasons(values, limits)
        values["balanced_450_reasons"] = ";".join(profile_reasons(values, PROFILES["balanced_450"]))
        rows.append(values)
        if i % 20000 == 0:
            print(
                json.dumps(
                    {
                        "described": i,
                        "total": len(source),
                        "seconds": round(time.monotonic() - start, 1),
                    }
                ),
                flush=True,
            )
    pq.write_table(
        pa.Table.from_pylist(rows),
        out / "per_graph_descriptors_private.parquet",
        compression="zstd",
    )
    refs = references(args.reference_library, args.review)
    for records in refs.values():
        for row in records:
            row["profile_reasons"] = {
                name: profile_reasons(row, limits) for name, limits in PROFILES.items()
            }
    dump(out / "reference_headgroups_private.json", refs)
    thresholds = (150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 750, 1000)
    n = len(rows)
    mass_cuts = []
    for limit in thresholds:
        count = sum(r["average_mass_Da"] <= limit for r in rows)
        mass_cuts.append({"upper_Da": limit, "count": count, "percent": 100 * count / n})
    mass_edges = [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 1500, 10000]
    hist, _ = np.histogram([r["average_mass_Da"] for r in rows], mass_edges)
    profiles = {}
    for name, limits in PROFILES.items():
        selected = [r for r in rows if r[name]]
        profiles[name] = {
            "limits": limits,
            "retained": len(selected),
            "percent": 100 * len(selected) / n,
            "family_counts": dict(Counter(r["chemistry_family"] for r in selected)),
            "overlapping_failure_reasons": dict(
                Counter(reason for r in rows for reason in profile_reasons(r, limits))
            ),
            "reference_coverage": {
                group: {
                    "kept": sum(not r["profile_reasons"][name] for r in records),
                    "total": len(records),
                }
                for group, records in refs.items()
            },
        }
    summary = {
        "schema": "polar-size-scope-audit-20261022",
        "graphs": n,
        "rdkit_version": rdBase.rdkitVersion,
        "source_sha256": before,
        "mass_definition": (
            "Existing model open Polar graph; mapped dummy mass=0; RDKit MolWt average mass, "
            "includes implicit/bracket H; excludes fixed glycerol and external fatty chains"
        ),
        "OH_capped_mass_offset_Da": 17.007,
        "descriptor_documentation": "https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html",
        "quantiles": {field: quantiles(rows, field) for field in METRICS},
        "family_counts": dict(Counter(r["chemistry_family"] for r in rows)),
        "origin_counts": dict(Counter(r["structure_origin"] for r in rows)),
        "historical_tier_counts": dict(Counter(r["polar_plausibility_tier"] for r in rows)),
        "family_quantiles": {
            family: {
                field: quantiles([r for r in rows if r["chemistry_family"] == family], field)
                for field in ("average_mass_Da", "C", "heavy_atoms", "smiles_tokens")
            }
            for family in sorted({r["chemistry_family"] for r in rows})
        },
        "mass_thresholds": mass_cuts,
        "mass_histogram": {"edges_Da": mass_edges, "counts": list(map(int, hist))},
        "element_presence": {
            e: sum(r[e] > 0 for r in rows) for e in ("C", "N", "O", "P", "S", "F", "Cl", "Br", "I")
        },
        "element_count_histograms": {
            e: dict(sorted(Counter(r[e] for r in rows).items()))
            for e in ("C", "N", "O", "P", "S", "F", "Cl", "Br", "I")
        },
        "has_any_halogen": sum(any(r[e] for e in ("F", "Cl", "Br", "I")) for r in rows),
        "outside_CHNOPS": sum("outside_CHNOPS" in chemistry_reasons(r) for r in rows),
        "chemistry_scope_failure_counts_overlap": dict(
            Counter(reason for r in rows for reason in chemistry_reasons(r))
        ),
        "chemistry_scope_pass": sum(not chemistry_reasons(r) for r in rows),
        "nitrogen_cut_sensitivity": {
            str(limit): sum(not chemistry_reasons(r, limit) for r in rows)
            for limit in (2, 3, 4, 5, 6, 8, 10)
        },
        "charged_atom_graphs": sum(r["charged_atom_count"] > 0 for r in rows),
        "net_charge_histogram": dict(Counter(r["formal_charge"] for r in rows)),
        "other_element_sets": dict(
            Counter(r["other_elements"] for r in rows if r["other_elements"])
        ),
        "extra_lipid_anchor_graphs": sum(r["extra_anchor"] for r in rows),
        "extra_lipid_anchor_reason_counts_overlap": dict(
            Counter(
                reason
                for r in rows
                if r["extra_anchor"]
                for reason in r["anchor_reason_codes"].split(";")
            )
        ),
        "profiles": profiles,
        "reference_mass_quantiles": {
            group: quantiles(records, "average_mass_Da") for group, records in refs.items()
        },
        "reference_only_mass_coverage": {
            group: {
                str(limit): sum(r["average_mass_Da"] <= limit for r in records)
                for limit in thresholds
            }
            for group, records in refs.items()
        },
        "formula_checks": {
            "all_parsed_and_single_mapped_P_boundary": True,
            "all_nonisotopic_average_masses_match_independent_element_H_counts": True,
            "radical_graphs": sum(r["radical_electrons"] > 0 for r in rows),
            "isotope_graphs": sum(r["isotope_atoms"] > 0 for r in rows),
        },
        "frozen_library_modified": False,
        "training_started": False,
        "selection_applied": False,
        "wall_seconds": round(time.monotonic() - start, 2),
    }
    assert digest(args.input) == before
    dump(out / "summary.json", summary)
    print(
        json.dumps(
            {"completed_graphs": n, "wall_seconds": summary["wall_seconds"], "output": str(out)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
