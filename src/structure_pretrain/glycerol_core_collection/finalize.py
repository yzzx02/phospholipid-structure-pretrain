"""Presentation and source QA without deriving new structures or template IDs."""

import gzip
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from .chemistry import canonical, core_definition
from .collection import dump_json, write_csv


def finalize(root):
    root = Path(root)
    rows = json.loads((root / "core_catalog_private.json").read_text(encoding="utf-8"))
    by_id = {r["core_id"]: r for r in rows}
    preferred_names = (
        "PC",
        "PE",
        "PS",
        "PA",
        "PG",
        "PI",
        "PIP",
        "PIP2",
        "PIP3",
        "MMPE",
        "DMPE",
        "Ala-PG",
        "Lys-PG",
        "MGDG",
        "DGDG",
        "SQDG",
        "3-SGDG",
        "DGTS",
        "DGTA",
        "DGCC",
        "GlcADG",
    )
    ambiguous = []
    for row in rows:
        preferred = [n for n in preferred_names if n in row["abbreviations"]]
        if preferred:
            row["display_name"] = "; ".join(preferred)
        row["display_chain_socket_count"] = row["core_smiles_chain_sites"].count("*")
        row["source_chain_occupancies"] = []
        positions = sorted(
            {e["name"] for e in row["natural_core_evidence"] if e["name"].startswith("PIP2[")}
        )
        row["source_explicit_inositol_position_labels"] = positions
        reason = "source_numbered_inositol_positions_share_stereo_free_graph_pending"
        if len(positions) > 1:
            if reason not in row["proposed_filter_reasons"]:
                row["proposed_filter_reasons"].append(reason)
            if row["selection_status"] == "结构检查通过":
                row["selection_status"] = "待判断"
            ambiguous.append(
                {
                    "core_id": row["core_id"],
                    "source_position_labels": positions,
                    "action": "Keep source labels. Do not silently promote as one position-resolved template; stereo-free graph alone cannot resolve the numbered symmetry.",  # noqa: E501
                }
            )
        if row["sources"] == ["PubChem_linkage_rule"]:
            donor_names = [x for x in row["names"] if "candidate from" not in x]
            row["display_name"] = "候选：" + "; ".join(donor_names[:3]) + " → 甘油糖苷"
    rows.sort(
        key=lambda r: (
            r["selection_status"] != "结构检查通过",
            not r["natural_core_verified"],
            r["P_count"] > 0,
            r["display_name"],
        )
    )
    summary = json.loads((root / "collection_summary.json").read_text(encoding="utf-8"))
    summary["selection_status_counts"] = dict(Counter(r["selection_status"] for r in rows))
    summary["numbered_position_ambiguity_core_count"] = len(ambiguous)
    summary["no_extra_stereochemical_ids"] = True
    summary["source_stereo_information_preserved_only_for_traceability"] = True
    summary["manual_keep_graphs_after_connectivity_merge"] = sum(
        "keep" in r["manual_decisions"] for r in rows
    )
    summary["prior_manual_exclude_graphs_preserved"] = sum(
        "exclude" in r["manual_decisions"] for r in rows
    )
    summary["prior_manual_pending_graphs_preserved"] = sum(
        "review" in r["manual_decisions"] for r in rows
    )
    checks = Counter()
    mismatches = []
    with gzip.open(root / "core_sources.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            extraction = source["extraction"]
            row = by_id[source["core_id"]]
            occupied = extraction.get("chain_count", 0)
            if occupied not in row["source_chain_occupancies"]:
                row["source_chain_occupancies"].append(occupied)
            if occupied > row["display_chain_socket_count"]:
                mol = Chem.MolFromSmiles(extraction["core_smiles_chain_sites"])
                Chem.RemoveStereochemistry(mol)
                definition = core_definition(canonical(mol))
                row.update(
                    core_smiles_chain_sites=definition["core_smiles_chain_sites"],
                    display_smiles=definition["core_display_smiles"],
                    P_first_status=definition["P_first_status"],
                    display_chain_socket_count=occupied,
                )
            if not source["full_lipid_with_chains"]:
                continue
            assert extraction["parent_roundtrip_formula_including_H"]
            checks["whole_lipid_records_connectivity_and_formula_roundtrip"] += 1
            checks["stereo_roundtrip_also_exact_but_not_template_selector"] += extraction[
                "parent_roundtrip_stereo"
            ]
            formula = source.get("source_formula", "").replace(" ", "")
            if formula:
                matched = formula == extraction["source_parent_formula"]
                checks[
                    "database_formula_equals_source_graph"
                    if matched
                    else "database_formula_differs_source_graph"
                ] += 1
                if not matched:
                    mismatches.append(
                        {
                            "source": source["source"],
                            "source_id": source["source_id"],
                            "source_url": source["source_url"],
                            "declared_formula": formula,
                            "graph_formula": extraction["source_parent_formula"],
                        }
                    )
    summary["source_structure_QA"] = dict(checks)
    summary["formula_mismatch_note"] = (
        "Source graph round-trip and database declared formula are distinct checks; mismatches remain explicit and are not repaired. Natural occurrence is never asserted per source acyl species."  # noqa: E501
    )
    dump_json(root / "core_catalog_private.json", rows)
    pq.write_table(
        pa.Table.from_pylist(rows), root / "glycerol_core_catalog.parquet", compression="zstd"
    )
    write_csv(root / "glycerol_core_catalog.csv", rows)
    dump_json(root / "collection_summary.json", summary)
    dump_json(root / "numbered_position_review_private.json", ambiguous)
    write_csv(root / "database_formula_mismatches.csv", mismatches)
    payload_path = root / "authoring/workbook_payload_private.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.update(cores=rows, summary=summary)
    dump_json(payload_path, payload)
    return summary
