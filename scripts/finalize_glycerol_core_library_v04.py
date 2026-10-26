"""Freeze a uniformly labelled glycerol/head library with explicit user overrides."""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
from freeze_combined_core_pool_v02 import polar_to_marked_core
from rdkit.Chem import rdMolDescriptors

from structure_pretrain.glycerol_core_collection.chemistry import (
    canonical,
    capped_core,
    parse,
    sugar_and_inositol,
)
from structure_pretrain.glycerol_core_collection.collection import dump_json, file_hash
from structure_pretrain.glycerol_substructure_collection.admission import MASS, readmission_group
from structure_pretrain.glycerol_substructure_collection.chemistry import (
    representations,
    role_frame,
    verify_assembly,
)

CONFIG = Path("configs/glycerol_core_unified_v04_20261026.json")
V2 = Path("outputs/template_guided_v2_20261023")


def check_labels(rep):
    mol = parse(rep["target_smiles"])
    assert canonical(mol, False) == rep["anchored_graph_key"]
    match, _, head_c, other = role_frame(mol)
    assert head_c != match[2], "sn2 head must not enter the frozen library"
    direct_p = mol.GetAtomWithIdx(other).GetAtomicNum() == 15
    assert bool(rep["rooted_at_attachment_P"]) == direct_p
    assert (other if direct_p else head_c) == 0, "SMILES must start at its defined root"
    gly = parse(rep["glycerol_first_smiles"])
    assert canonical(gly, False) == rep["anchored_graph_key"]
    assert role_frame(gly)[2] == 0
    return mol


def normalize_saved_schema():
    config = json.loads(CONFIG.read_text("utf-8"))
    root = Path(config["root"])
    path = root / "library_manifest_v04.json"
    manifest = json.loads(path.read_text("utf-8"))
    schema = pq.read_schema(manifest["files"]["main_core_library_v04"]["path"])
    for record in manifest["files"].values():
        table_path = Path(record["path"])
        assert file_hash(table_path) == record["sha256"]
        table = pq.read_table(table_path)
        assert table.num_rows == record["rows"]
        if not table.schema.equals(schema):
            table = table.cast(schema)
            pq.write_table(table, table_path.with_suffix(".parquet.part"), compression="zstd")
            table_path.with_suffix(".parquet.part").replace(table_path)
            record.update(sha256=file_hash(table_path), bytes=table_path.stat().st_size)
        assert pq.read_schema(table_path).equals(schema)
    manifest.update(all_Parquet_schemas_identical=True, schema=str(schema))
    dump_json(path, manifest)
    print(json.dumps({"all_Parquet_schemas_identical": True, "main": manifest["main_count"]}))


def main():
    config = json.loads(CONFIG.read_text("utf-8"))
    root, prior = Path(config["root"]), Path(config["prior_root"])
    root.mkdir(parents=True, exist_ok=True)
    original_path = Path(config["original_library"])
    prior_manifest = json.loads((prior / "collection_summary_v03.json").read_text("utf-8"))
    inputs = [
        CONFIG,
        original_path,
        prior / "combined_candidate_cores_v03.parquet",
        prior / "phosphate_scope_reaudit.parquet",
        prior / "standby_43100_mass_shift.parquet",
        prior / "observed_source_mapping.jsonl.gz",
        Path("outputs/headgroup_training_targets_v02_20261026/training_targets.parquet"),
    ]
    protected = {str(p): file_hash(p) for p in inputs}
    assert (
        protected[str(original_path)] == prior_manifest["old_reaudit"]["original_SHA256_unchanged"]
    )
    assert (
        protected[str(prior / "combined_candidate_cores_v03.parquet")]
        == prior_manifest["combined_parquet_SHA256"]
    )
    contract = json.loads((V2 / "queue_started.json").read_text("utf-8"))["contract"]
    assert all(file_hash(p) == h for p, h in contract["source_sha256"].items())
    assert file_hash(V2 / "data/manifest.json") == contract["data_manifest_sha256"]
    audit_rows = pq.read_table(prior / "phosphate_scope_reaudit.parquet").to_pylist()
    audit = {r["polar_id"]: r for r in audit_rows}
    original = {r["polar_id"]: r for r in pq.read_table(original_path).to_pylist()}
    reference = {r["target_id"]: r for r in pq.read_table(inputs[-1]).to_pylist()}
    standby_cores = {
        r["polar_id"]: r["marked_core_smiles"]
        for r in pq.read_table(prior / "standby_43100_mass_shift.parquet").to_pylist()
    }
    prior_rows = pq.read_table(prior / "combined_candidate_cores_v03.parquet").to_pylist()
    main_rows, standby_rows, excluded_rows = [], [], []
    counters, seen, prior_background = Counter(), set(), set()
    source_refs = {
        "original_P_records": str(original_path),
        "prior_reference_and_rule_records": str(inputs[-1]),
        "PubChem_source_edges": str(prior / "observed_source_mapping.jsonl.gz"),
        "PubChem_source_records": str(prior / "raw/nonP_properties"),
    }

    def uniform(
        rep,
        *,
        old=None,
        prior_row=None,
        flags=(),
        status="main",
        basis="v03_retained",
        explicit_proof=False,
    ):
        mol = check_labels(rep)
        if explicit_proof:
            proof = verify_assembly(mol)
            formula, mass = proof["core_formula"], proof["core_average_mass_Da"]
            counters["newly_admitted_explicit_assembly_checks"] += 1
        elif prior_row is not None:
            assert prior_row["full_assembly_H_formula_pass"]
            formula, mass = prior_row["core_formula"], prior_row["core_average_mass_Da"]
        else:
            formula = rdMolDescriptors.CalcMolFormula(capped_core(mol))
            mass = audit[old["polar_id"]]["core_average_mass_Da_recomputed"]
        sugar, inositol = sugar_and_inositol(mol)
        limit = 600.0 if sugar or inositol else 500.0
        assert set(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum()) <= set(
            config["allowed_elements"]
        )
        p_count = sum(a.GetAtomicNum() == 15 for a in mol.GetAtoms())
        bg = prior_row.get("background_polar_ids", []) if prior_row else [old["polar_id"]]
        meta = reference.get(rep["target_id"], {})
        raw_name = (prior_row or {}).get("display_name", "") or meta.get("display_name", "")
        source_origins = sorted({original[k]["structure_origin"] for k in bg})
        evidence = (
            "database_glycerol_substructure"
            if (prior_row or {}).get("observed_substructure_evidence")
            else "prior_reference_or_rule_record"
            if meta
            else "original_P_source_record"
        )
        row = {
            "target_id": rep["target_id"],
            "core_scope": "glycerol_plus_head",
            "selection_status": status,
            "admission_basis": basis,
            "fixed_glycerol_backbone_present": True,
            "head_attachment_position": "terminal",
            "chain_attachment_roles": "101:terminal_glycerol_O;102:central_glycerol_O",
            "contains_P": bool(p_count),
            "P_count": p_count,
            "rooted_at_attachment_P": bool(rep["rooted_at_attachment_P"]),
            "target_smiles": rep["target_smiles"],
            "glycerol_first_smiles": rep["glycerol_first_smiles"],
            "capped_core_smiles": rep["capped_core_smiles"],
            "legacy_Polar_fragment_smiles": rep["polar_P_first_smiles"],
            "anchored_graph_key": rep["anchored_graph_key"],
            "core_formula": formula,
            "core_average_mass_Da": float(mass),
            "nominal_mass_limit_Da": limit,
            "exceeds_nominal_mass_limit": mass > limit + 1e-8,
            "user_mass_exception": basis == "user_accepted_mass_only_exception",
            "sugar_ring_count": sugar,
            "inositol_ring_count": inositol,
            "screening_flags": sorted(flags),
            "risk_flags": sorted(set(flags) - {MASS}),
            "source_origins": source_origins,
            "display_name": raw_name or rep["target_id"],
            "name_status": "source_name" if raw_name else "stable_internal_ID",
            "aliases": meta.get("names", []),
            "source_urls": meta.get("source_urls", []),
            "background_polar_ids": bg,
            "PubChem_source_CIDs_sample": (prior_row or {}).get("PubChem_source_CIDs", []),
            "prior_expansion_target_id": (prior_row or {}).get("expansion_target_id", ""),
            "evidence_basis": evidence,
            "observed_substructure_evidence": bool(
                (prior_row or {}).get("observed_substructure_evidence")
            ),
            "source_uncut_core_observed": bool((prior_row or {}).get("source_uncut_core_observed")),
            "assembly_H_formula_verified": True,
            "assembly_proof_basis": "new_reference_chain_reassembly"
            if explicit_proof
            else "unchanged_hash_bound_prior_proof",
            "source_stereochemistry_preserved_in_originals": True,
            "stereo_ignored_for_target_identity": True,
            "used_in_current_V2_training": False,
        }
        if basis == "user_accepted_mass_only_exception":
            assert row["exceeds_nominal_mass_limit"] and set(flags) == {MASS}
        assert row["target_id"] not in seen
        seen.add(row["target_id"])
        counters[status] += 1
        counters["all_root_port_graph_checks"] += 1
        return row

    for i, oldrow in enumerate(prior_rows, 1):
        prior_background.update(oldrow["background_polar_ids"])
        main_rows.append(uniform(oldrow, prior_row=oldrow))
        if i % 16384 == 0:
            print(json.dumps({"prior_cores_labelled": i}), flush=True)
    added = Counter()
    for i, old in enumerate(original.values(), 1):
        ident = old["polar_id"]
        if ident in prior_background:
            continue
        review = audit[ident]
        flags = set(filter(None, review["new_flags"].split(";")))
        group = readmission_group(review)
        mol = (
            parse(standby_cores[ident])
            if ident in standby_cores
            else polar_to_marked_core(old["polar_fragment_smiles"])
        )
        rep = representations(mol)
        if group:
            main_rows.append(uniform(rep, old=old, flags=flags, basis=group, explicit_proof=True))
            added[group] += 1
        elif review["mass_shift_standby"]:
            standby_rows.append(
                uniform(
                    rep,
                    old=old,
                    flags=flags,
                    status="standby",
                    basis="mass_plus_other_flags_not_reinstated",
                )
            )
        else:
            excluded_rows.append(
                uniform(
                    rep,
                    old=old,
                    flags=flags,
                    status="excluded_or_review",
                    basis="remaining_scope_exclusion",
                )
            )
        if i % 16384 == 0:
            print(
                json.dumps({"original_records_visited": i, "new_admissions": sum(added.values())}),
                flush=True,
            )
    assert added["user_accepted_mass_only_exception"] == config["admit_mass_only_standby_count"]
    assert added["user_accepted_risk_annotations"] == config["admit_risk_only_count"]
    assert len(main_rows) == 189864 and len(standby_rows) == 5491 and len(excluded_rows) == 31132
    assert sum(not r["contains_P"] for r in main_rows) == config["fixed_nonP_count"]
    old_np = {r["target_id"] for r in prior_rows if r["P_count"] == 0}
    assert old_np == {r["target_id"] for r in main_rows if not r["contains_P"]}
    files = {}
    shared_schema = pa.Table.from_pylist(main_rows).schema
    for name, rows in (
        ("main_core_library_v04", main_rows),
        ("standby_core_library_v04", standby_rows),
        ("excluded_core_library_v04", excluded_rows),
    ):
        rows.sort(key=lambda r: r["target_id"])
        path = root / (name + ".parquet")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=shared_schema),
            path.with_suffix(".parquet.part"),
            compression="zstd",
        )
        path.with_suffix(".parquet.part").replace(path)
        files[name] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
    assert all(file_hash(p) == h for p, h in protected.items())
    assert all(file_hash(p) == h for p, h in contract["source_sha256"].items())
    report = {
        "client_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "main_count": len(main_rows),
        "main_P_count": sum(r["contains_P"] for r in main_rows),
        "main_nonP_count": len(old_np),
        "readmission_counts": dict(added),
        "remaining_standby": len(standby_rows),
        "remaining_excluded_or_review": len(excluded_rows),
        "all_status_count": len(seen),
        "all_core_scope": "glycerol_plus_head",
        "nonP_ID_set_frozen_exactly": True,
        "no_new_enumeration_or_database_search": True,
        "all_target_smiles_include_fixed_glycerol": True,
        "labels_and_H_formula_proof_counts": dict(counters),
        "risk_annotations_preserved": True,
        "nominal_mass_limits_unchanged_except_explicit_admitted_IDs": True,
        "input_SHA256_unchanged": protected,
        "protected_V2_source_files_unchanged": len(contract["source_sha256"]),
        "source_references": source_refs,
        "files": files,
        "current_V2_inputs_changed": False,
        "new_library_training_started": False,
        "default_future_training_candidates": files["main_core_library_v04"]["path"],
    }
    dump_json(root / "library_manifest_v04.json", report)
    dump_json(Path("docs/results/GLYCEROL_CORE_UNIFIED_V04_20261026.json"), report)
    print(
        json.dumps(
            {
                "complete": True,
                "main": len(main_rows),
                "P": report["main_P_count"],
                "nonP": len(old_np),
                "remaining_standby": len(standby_rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize-schema-only", action="store_true")
    args = parser.parse_args()
    if args.normalize_schema_only:
        normalize_saved_schema()
    else:
        main()
        normalize_saved_schema()
