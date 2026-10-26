"""Extract all declared nonP source records and merge with protected prior pools."""

import gzip
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from structure_pretrain.glycerol_core_collection.chemistry import parse
from structure_pretrain.glycerol_core_collection.collection import dump_json, file_hash
from structure_pretrain.glycerol_substructure_collection.chemistry import (
    extract_observed_cores,
    policy,
    representations,
    verify_assembly,
)

CONFIG = Path("configs/glycerol_substructure_collection_v03_20261026.json")
ROOT = Path("outputs/glycerol_core_collection_v03_20261026")
BASE = Path("outputs/glycerol_core_training_pool_v02_20261026")
ELEMENTS = {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"}


def process_batch(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = json.load(stream)
    accepted, rejected, counts = [], [], Counter()
    for source in records:
        cid = int(source["CID"])
        counts["source_records"] += 1
        # Element restrictions apply to the retained target, not clipped groups/counterions.
        try:
            candidates, statuses = extract_observed_cores(
                source["SMILES"], allow_other_site_substructure_cuts=True
            )
        except (ValueError, RuntimeError) as error:
            rejected.append(
                (cid, "source", type(error).__name__ + ":" + str(error)[:100], "", "", 0.0)
            )
            continue
        if not candidates:
            rejected.append((cid, "source", ";".join(sorted(statuses)), "", "", 0.0))
        source_accepted = False
        for item in candidates:
            mol = item.pop("mol")
            try:
                desc = policy(mol, new_pubchem_branch=True)
                rep = representations(mol)
            except (ValueError, RuntimeError) as error:
                rejected.append(
                    (
                        cid,
                        "core",
                        "core_validation:" + type(error).__name__ + ":" + str(error)[:100],
                        "",
                        "",
                        0.0,
                    )
                )
                continue
            if desc["filter_reasons"]:
                rejected.append(
                    (
                        cid,
                        "core",
                        ";".join(desc["filter_reasons"]),
                        rep["target_id"],
                        rep["head_attachment_position"],
                        desc["core_average_mass_Da"],
                    )
                )
                continue
            assert desc["P_count"] == 0 and rep["head_attachment_position"] == "terminal"
            accepted.append(
                {
                    **rep,
                    "core_formula": desc["core_formula"],
                    "core_average_mass_Da": desc["core_average_mass_Da"],
                    "P_count": 0,
                    "halogen_count": desc["halogen_count"],
                    "source_CID": cid,
                    "source_name": source.get("IUPACName", ""),
                    **item,
                }
            )
            source_accepted = True
        counts["accepted_source_CIDs"] += source_accepted
    return accepted, rejected, dict(counts), [int(r["CID"]) for r in records]


def main():
    config = json.loads(CONFIG.read_text("utf-8"))
    assert config["new_PubChem_branch_requires_no_P"]
    assert config["new_PubChem_branch_requires_terminal_head_attachment"]
    fetched = json.loads((ROOT / "raw/fetch_complete.json").read_text("utf-8"))
    assert fetched["properties_complete"] and fetched["properties_count"] == 480173
    base_path = BASE / "combined_candidate_cores.parquet"
    base_manifest = json.loads((BASE / "freeze_manifest.json").read_text("utf-8"))
    protected = {
        str(base_path): file_hash(base_path),
        "data/structure_library_v0.3/polar_library.parquet": file_hash(
            "data/structure_library_v0.3/polar_library.parquet"
        ),
        "outputs/headgroup_training_targets_v02_20261026/training_targets.parquet": file_hash(
            "outputs/headgroup_training_targets_v02_20261026/training_targets.parquet"
        ),
    }
    assert protected[str(base_path)] == base_manifest["combined_parquet_SHA256"]
    rows = pq.read_table(base_path).to_pylist()
    for row in rows:
        row.update(
            PubChem_source_CIDs=[],
            observed_substructure_evidence=False,
            head_attachment_position="terminal",
            legacy_terminal_frame_compatible=True,
            source_without_fatty_chains_observed=False,
            source_uncut_core_observed=False,
            source_without_nonfatty_cuts_observed=False,
        )
    by_key = {r["anchored_graph_key"]: r for r in rows}
    before_nonp = sum(r["P_count"] == 0 for r in by_key.values())
    totals, reject_counts, rejected_examples = Counter(), Counter(), defaultdict(list)
    seen_cids, known_new_keys = set(), set()
    rejection_writer = None
    provenance = gzip.open(ROOT / "observed_source_mapping.jsonl.gz", "wt", encoding="utf-8")
    paths = sorted((ROOT / "raw/nonP_properties").glob("properties_*.json.gz"))
    with ProcessPoolExecutor(max_workers=2) as executor:
        for accepted, rejected, counts, cids in executor.map(process_batch, paths, chunksize=1):
            assert not seen_cids.intersection(cids)
            seen_cids.update(cids)
            totals.update(counts)
            for item in accepted:
                cid, name = item.pop("source_CID"), item.pop("source_name")
                key = item["anchored_graph_key"]
                if key not in by_key:
                    proof = verify_assembly(parse(item["target_smiles"]))
                    assert proof["core_formula"] == item["core_formula"]
                    by_key[key] = {
                        **item,
                        "background_polar_ids": [],
                        "expansion_target_id": "",
                        "origin_category": "observed_PubChem_nonP_terminal_core",
                        "display_name": name
                        if not item["source_cut_substituent_count"]
                        and not item.get("source_parent_had_multiple_components", False)
                        else "Observed core from PubChem CID" + str(cid),
                        "new_split": "unassigned",
                        "full_assembly_H_formula_pass": True,
                        "PubChem_source_CIDs": [],
                        "observed_substructure_evidence": True,
                        "source_without_fatty_chains_observed": False,
                        "source_uncut_core_observed": False,
                        "source_without_nonfatty_cuts_observed": False,
                    }
                    known_new_keys.add(key)
                row = by_key[key]
                if len(row["PubChem_source_CIDs"]) < 12:
                    row["PubChem_source_CIDs"].append(cid)
                row["observed_substructure_evidence"] = True
                row["source_roundtrip_H_formula_pass"] = True
                row["source_without_fatty_chains_observed"] |= not item[
                    "observed_fatty_chain_count"
                ]
                row["source_uncut_core_observed"] |= not item["source_cut_substituent_count"]
                row["source_without_nonfatty_cuts_observed"] |= not item[
                    "source_nonfatty_substituent_cut_count"
                ]
                edge = {
                    "source_CID": cid,
                    "target_id": item["target_id"],
                    "source_name": name,
                    "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/" + str(cid),
                    "observed_fatty_chain_count": item["observed_fatty_chain_count"],
                    "source_cut_substituent_count": item["source_cut_substituent_count"],
                    "source_nonfatty_substituent_cut_count": item[
                        "source_nonfatty_substituent_cut_count"
                    ],
                    "source_glycerol_atom_indices": item["source_glycerol_atom_indices"],
                    "source_head_O_atom_index": item["source_head_O_atom_index"],
                    "source_head_neighbor_atom_index": item["source_head_neighbor_atom_index"],
                    "source_component_smiles": item.get("source_component_smiles", ""),
                    "source_roundtrip_scope": "connected_source_component",
                    "source_graph_and_H_formula_roundtrip": True,
                    "evidence_kind": "database_observed_substructure_not_natural_occurrence_proof",
                }
                provenance.write(json.dumps(edge, ensure_ascii=False) + "\n")
                totals["accepted_source_core_edges"] += 1
            rejected_rows = []
            for cid, kind, why, target, position, mass in rejected:
                for reason in why.split(";"):
                    reject_counts[reason] += 1
                    if len(rejected_examples[reason]) < 12:
                        rejected_examples[reason].append(
                            {
                                "CID": cid,
                                "target_id": target,
                                "head_position": position,
                                "core_mass_Da": mass,
                            }
                        )
                rejected_rows.append(
                    {
                        "CID": cid,
                        "record_kind": kind,
                        "reasons": why,
                        "target_id": target,
                        "head_position": position,
                        "core_mass_Da": mass,
                    }
                )
            if rejected_rows:
                table = pa.Table.from_pylist(rejected_rows)
                if rejection_writer is None:
                    rejection_writer = pq.ParquetWriter(
                        ROOT / "pending_source_or_core.parquet.part",
                        table.schema,
                        compression="zstd",
                    )
                rejection_writer.write_table(table)
            if totals["source_records"] % 25000 == 0:
                print(
                    json.dumps(
                        {
                            "processed_nonP_sources": totals["source_records"],
                            "new_unique_cores": len(known_new_keys),
                        }
                    ),
                    flush=True,
                )
    provenance.close()
    if rejection_writer:
        rejection_writer.close()
        (ROOT / "pending_source_or_core.parquet.part").replace(
            ROOT / "pending_source_or_core.parquet"
        )
    expected = np.load(ROOT / "raw/nonP_scaffold_cids.npy")
    assert seen_cids == set(expected.tolist()) and totals["source_records"] == 480173
    old = json.loads((ROOT / "phosphate_scope_reaudit_summary.json").read_text("utf-8"))
    recovered = pq.read_table(ROOT / "old_recovered_cores.parquet").to_pylist()
    assert len(recovered) == old["newly_readmitted_total"]
    for row in recovered:
        assert row["anchored_graph_key"] not in by_key and row["P_count"] > 0
        row.update(
            PubChem_source_CIDs=[],
            observed_substructure_evidence=False,
            source_without_fatty_chains_observed=False,
            source_uncut_core_observed=False,
            source_without_nonfatty_cuts_observed=False,
        )
        by_key[row["anchored_graph_key"]] = row
    final = sorted(by_key.values(), key=lambda r: r["target_id"])
    for row in final:
        formula = dict(re.findall(r"([A-Z][a-z]?)(\d*)", row["core_formula"]))
        row["halogen_count"] = sum(
            int(formula[e] or 1) for e in ("F", "Cl", "Br", "I") if e in formula
        )
        row.setdefault("source_roundtrip_H_formula_pass", None)
    assert len({r["target_id"] for r in final}) == len(final)
    assert all(r["head_attachment_position"] == "terminal" for r in final)
    assert all(file_hash(p) == h for p, h in protected.items())
    pq.write_table(
        pa.Table.from_pylist(final),
        ROOT / "combined_candidate_cores_v03.parquet",
        compression="zstd",
    )
    summary = {
        "source_query_count": 833819,
        "P_source_CIDs_removed_from_new_branch": 353646,
        "nonP_source_CIDs_processed": len(seen_cids),
        "source_counts": dict(totals),
        "previous_nonP_cores": before_nonp,
        "new_unique_nonP_cores": len(known_new_keys),
        "final_nonP_cores": sum(r["P_count"] == 0 for r in final),
        "final_P_cores": sum(r["P_count"] > 0 for r in final),
        "final_core_count": len(final),
        "nonP_graphs_with_observed_database_substructure": sum(
            r["P_count"] == 0 and r["observed_substructure_evidence"] for r in final
        ),
        "nonP_graphs_observed_without_fatty_chains": sum(
            r["P_count"] == 0 and r["source_without_fatty_chains_observed"] for r in final
        ),
        "nonP_graphs_observed_as_uncut_whole_cores": sum(
            r["P_count"] == 0 and r["source_uncut_core_observed"] for r in final
        ),
        "nonP_graphs_observed_without_nonfatty_site_cuts": sum(
            r["P_count"] == 0 and r["source_without_nonfatty_cuts_observed"] for r in final
        ),
        "substructure_cut_mode": (
            "other glycerol O substituents may be explicitly cut; entire head preserved"
        ),
        "processing_config_SHA256": file_hash(CONFIG),
        "element_restriction_scope": "retained_target_core",
        "new_branch_P_or_sn2_cores_included": 0,
        "rejection_counts_are_events_and_can_overlap": dict(reject_counts),
        "old_reaudit": old,
        "protected_input_SHA256_unchanged": protected,
        "combined_parquet_SHA256": file_hash(ROOT / "combined_candidate_cores_v03.parquet"),
        "full_assembly_proof_count": len(final),
        "original_library_modified": False,
        "current_V2_inputs_changed": False,
        "new_pool_training_started": False,
        "complete_only_for_declared_acyclic_glycerol_scaffold_query": True,
        "does_not_claim_all_possible_glycerol_lipid_chemistry_exhausted": True,
    }
    dump_json(ROOT / "collection_summary_v03.json", summary)
    dump_json(ROOT / "rejection_examples_private.json", dict(rejected_examples))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
