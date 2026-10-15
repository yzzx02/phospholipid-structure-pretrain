import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from .chemistry import (
    Reject,
    acidic_sites,
    process_smiles,
    read_controls,
    select_component,
    validate_controls,
)
from .client import PubChemClient, atomic_json, run_query


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


class ParquetSink:
    def __init__(self, path, schema):
        self.path, self.schema, self.rows = Path(path), schema, []
        self.temporary = self.path.with_suffix(".parquet.tmp")
        self.writer = pq.ParquetWriter(self.temporary, schema, compression="zstd")

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= 1000:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()
        self.temporary.replace(self.path)


def schema(strings=(), integers=(), floats=(), booleans=(), lists=(), cid_lists=()):
    return pa.schema(
        [(s, pa.string()) for s in strings]
        + [(s, pa.int64()) for s in integers]
        + [(s, pa.float64()) for s in floats]
        + [(s, pa.bool_()) for s in booleans]
        + [(s, pa.list_(pa.string())) for s in lists]
        + [(s, pa.list_(pa.int64())) for s in cid_lists]
    )


RAW_SCHEMA = schema(
    strings=[
        "canonical_smiles",
        "isomeric_smiles",
        "original_smiles",
        "molecular_formula",
        "inchi_key",
        "title",
    ],
    integers=["cid", "formal_charge"],
    floats=["molecular_weight"],
    lists=["matched_query_ids", "matched_families"],
)
COMPONENT_SCHEMA = schema(
    strings=["original_smiles", "selected_p_component"],
    integers=[
        "cid",
        "source_component_charge",
        "attachment_site_count",
        "distinct_attachment_fragments",
    ],
    lists=["removed_counterions"],
)
CANDIDATE_SCHEMA = schema(
    strings=[
        "polar_family",
        "polar_fragment_smiles",
        "standardized_fragment_smiles",
        "source_charge_form",
        "molecular_formula",
        "status",
    ],
    integers=[
        "cid",
        "source_attachment_atom_index",
        "attachment_site_count_before_dedup",
        "distinct_attachment_fragments",
        "carbon_count",
        "phosphorus_count",
        "formal_charge",
        "nitrogen_count",
        "sulfur_count",
        "heavy_atoms",
        "max_aliphatic_run",
    ],
    floats=["molecular_weight"],
    booleans=["charge_ambiguous", "ring_containing"],
    lists=["charge_alternatives", "filter_flags", "matched_query_ids"],
)
FRAGMENT_SCHEMA = schema(
    strings=["polar_id", "polar_family", "polar_fragment_smiles", "molecular_formula", "status"],
    integers=[
        "source_count",
        "attachment_site_count_before_dedup",
        "carbon_count",
        "phosphorus_count",
        "formal_charge",
        "nitrogen_count",
        "sulfur_count",
        "heavy_atoms",
        "max_aliphatic_run",
    ],
    floats=["molecular_weight"],
    booleans=["ring_containing", "allow_aa", "allow_oa", "allow_pa"],
    lists=["matched_query_ids", "known_control_ids", "review_flags", "filter_flags"],
    cid_lists=["source_cids"],
)
REJECTION_SCHEMA = schema(
    strings=["original_smiles", "reason", "detail"], integers=["cid"], lists=["matched_query_ids"]
)


def live_controls(client, config, controls, known_keys):
    result = {}
    for control in controls:
        name, control_id = control["pubchem_name"], control["control_id"]
        if not name:
            result[control_id] = {"local_only": True, "reason": "synthetic metal negative"}
            continue
        rows = client.properties_by_name(name)
        if not rows:
            raise RuntimeError(f"PubChem control name not resolved: {name}")
        checks = []
        for row in rows:
            try:
                processed = process_smiles(
                    row.get("SMILES") or row["ConnectivitySMILES"], config.get("limits")
                )
                keys = {c["polar_fragment_smiles"] for c in processed["candidates"]}
                passed = (
                    control["kind"] == "positive"
                    and bool(keys & set(known_keys[control_id]))
                    and not any(c["charge_ambiguous"] for c in processed["candidates"])
                )
                checks.append({"cid": row["CID"], "passed": passed, "fragment_keys": sorted(keys)})
            except Reject as exc:
                checks.append(
                    {
                        "cid": row["CID"],
                        "passed": control["kind"] == "negative",
                        "reason": exc.reason,
                    }
                )
        result[control_id] = {"passed": all(c["passed"] for c in checks), "records": checks}
        log(f"live control {control_id}: {result[control_id]['passed']}")
    if not all(r.get("passed", r.get("local_only", False)) for r in result.values()):
        raise RuntimeError(f"Live controls failed; harvest stopped: {result}")
    return result


def normalize_raw(row, query_ids, query_families):
    return {
        "cid": row["CID"],
        "canonical_smiles": row.get("ConnectivitySMILES", ""),
        "isomeric_smiles": row.get("SMILES", ""),
        "original_smiles": row.get("SMILES") or row.get("ConnectivitySMILES", ""),
        "molecular_formula": row.get("MolecularFormula", ""),
        "molecular_weight": float(row["MolecularWeight"]),
        "formal_charge": row["Charge"],
        "inchi_key": row.get("InChIKey", ""),
        "title": row.get("Title", ""),
        "matched_query_ids": sorted(query_ids),
        "matched_families": sorted(query_families),
    }


STATUS_PRIORITY = {
    "APPROVABLE": 0,
    "MULTIPLE_ATTACHMENT_AMBIGUITY": 1,
    "FILTER_REVIEW": 2,
    "CHARGE_AMBIGUOUS": 3,
}


def property_batches(client, cids, batch_size):
    """Overlap one network worker with CPU chemistry; never parallelize HTTP requests."""
    if not cids:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pubchem-properties") as pool:
        future = pool.submit(client.get_properties, cids[:batch_size])
        for start in range(0, len(cids), batch_size):
            batch = cids[start : start + batch_size]
            try:
                properties, error = future.result(), None
            except Exception as exc:
                properties, error = [], exc
            next_start = start + batch_size
            if next_start < len(cids):
                future = pool.submit(
                    client.get_properties, cids[next_start : next_start + batch_size]
                )
            yield start, batch, properties, error


def merge_candidate(fragments, candidate, cid, query_ids, known_keys):
    key = candidate["polar_fragment_smiles"]
    if key not in fragments:
        row = {k: candidate.get(k) for k in FRAGMENT_SCHEMA.names}
        row.update(
            {
                "polar_id": "PCP_" + hashlib.sha256(key.encode()).hexdigest()[:16],
                "source_cids": set(),
                "matched_query_ids": set(),
                "review_flags": set(),
                "filter_flags": set(),
                "attachment_site_count_before_dedup": 0,
                "known_control_ids": [name for name, keys in known_keys.items() if key in keys],
                "allow_aa": True,
                "allow_oa": True,
                "allow_pa": key in known_keys.get("PE", []),
            }
        )
        fragments[key] = row
    row = fragments[key]
    row["source_cids"].add(cid)
    row["matched_query_ids"].update(query_ids)
    row["review_flags"].add(candidate["status"])
    row["filter_flags"].update(candidate["filter_flags"])
    row["attachment_site_count_before_dedup"] += 1
    row["status"] = max(row["status"], candidate["status"], key=STATUS_PRIORITY.get)


def distribution(values):
    values = sorted(values)
    if not values:
        return {}
    return {
        "min": values[0],
        "median": values[len(values) // 2],
        "p95": values[int((len(values) - 1) * 0.95)],
        "max": values[-1],
    }


def process_records(client, matches, queries, output, config, known_keys):
    sinks = {
        name: ParquetSink(output / f"pubchem_{name}.parquet", spec)
        for name, spec in (
            ("polar_raw", RAW_SCHEMA),
            ("p_components", COMPONENT_SCHEMA),
            ("attachment_candidates", CANDIDATE_SCHEMA),
            ("polar_rejections", REJECTION_SCHEMA),
        )
    }
    fragments, reasons, stats = {}, Counter(), Counter()
    stats.update(
        {
            name: 0
            for name in (
                "properties_downloaded",
                "single_p_component_records",
                "counterion_records",
                "counterions_removed",
                "attachment_eligible_cids",
                "attachment_candidates",
                "unique_cid_fragment_pairs",
            )
        }
    )
    families = {q["query_id"]: q["family"] for q in queries}
    cids = sorted(matches)
    batch_size = config.get("api", {}).get("property_batch_size", 100)
    missing_cids = []
    for start, batch, properties, error in property_batches(client, cids, batch_size):
        if error is not None:
            # Preserve all unprocessed CIDs; do not misclassify network failures as chemistry.
            missing_cids.extend(batch)
            log(f"properties batch failed ({len(batch)} CIDs): {error}")
            continue
        for prop in properties:
            cid = prop["CID"]
            row = normalize_raw(prop, matches[cid], {families[q] for q in matches[cid]})
            sinks["polar_raw"].add(row)
            stats["properties_downloaded"] += 1
            component_row = None
            try:
                selected, removed = select_component(row["original_smiles"])
                component_row = {
                    "cid": cid,
                    "original_smiles": row["original_smiles"],
                    "selected_p_component": Chem.MolToSmiles(selected),
                    "removed_counterions": removed,
                    "source_component_charge": Chem.GetFormalCharge(selected),
                    "attachment_site_count": len(acidic_sites(selected)),
                }
                stats["single_p_component_records"] += 1
                stats["counterion_records"] += bool(removed)
                stats["counterions_removed"] += len(removed)
                processed = process_smiles(
                    row["original_smiles"], config.get("limits"), selected=(selected, removed)
                )
                component_row["distinct_attachment_fragments"] = processed[
                    "distinct_attachment_fragments"
                ]
                stats["attachment_eligible_cids"] += 1
                stats["attachment_candidates"] += len(processed["candidates"])
                stats["unique_cid_fragment_pairs"] += processed["distinct_attachment_fragments"]
                for candidate in processed["candidates"]:
                    sinks["attachment_candidates"].add(
                        {"cid": cid, **candidate, "matched_query_ids": sorted(matches[cid])}
                    )
                    merge_candidate(fragments, candidate, cid, matches[cid], known_keys)
            except Reject as exc:
                reasons[exc.reason] += 1
                sinks["polar_rejections"].add(
                    {
                        "cid": cid,
                        "original_smiles": row["original_smiles"],
                        "reason": exc.reason,
                        "detail": str(exc),
                        "matched_query_ids": sorted(matches[cid]),
                    }
                )
            finally:
                if component_row is not None:
                    sinks["p_components"].add(component_row)
        if start % 1000 == 0:
            log(
                f"processed {min(start + batch_size, len(cids))}/{len(cids)} CIDs; "
                f"{len(fragments)} fragments"
            )
            atomic_json(
                output / "checkpoint.json",
                {
                    "last_batch_start": start,
                    "processed": dict(stats),
                    "unique_fragments_so_far": len(fragments),
                    "resume": "Rerun same command; cached API/properties are reused.",
                },
            )
    for sink in sinks.values():
        sink.close()
    fragment_rows = []
    for row in sorted(fragments.values(), key=lambda r: r["polar_id"]):
        for name in ("source_cids", "matched_query_ids", "review_flags", "filter_flags"):
            row[name] = sorted(row[name])
        row["source_count"] = len(row["source_cids"])
        fragment_rows.append(row)
    pq.write_table(
        pa.Table.from_pylist(fragment_rows, schema=FRAGMENT_SCHEMA),
        output / "pubchem_polar_fragments.parquet",
        compression="zstd",
    )
    with (output / "pubchem_polar_review.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAGMENT_SCHEMA.names, delimiter="\t")
        writer.writeheader()
        for row in fragment_rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                    for k, v in row.items()
                }
            )
    family_counts = Counter(r["polar_family"] for r in fragment_rows)
    status_counts = Counter(r["status"] for r in fragment_rows)
    representatives = []
    # Round-robin families, showing actual provenance, excluding all controls.
    buckets = {
        f: sorted(
            [
                r
                for r in fragment_rows
                if r["polar_family"] == f
                and not r["known_control_ids"]
                and r["status"] == "APPROVABLE"
            ],
            key=lambda r: (r["heavy_atoms"], r["polar_id"]),
        )
        for f in family_counts
    }
    while len(representatives) < 20 and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(representatives) < 20:
                representatives.append(bucket.pop(0))
    pairs = stats["unique_cid_fragment_pairs"]
    report = {
        **dict(stats),
        "unique_fragments": len(fragment_rows),
        "family_counts": dict(family_counts),
        "status_counts": dict(status_counts),
        "rejections": dict(reasons),
        "missing_property_cids": missing_cids,
        "cid_fragment_pair_dedup_rate": 1 - len(fragment_rows) / pairs if pairs else 0,
        "attachment_candidate_dedup_rate": 1 - len(fragment_rows) / stats["attachment_candidates"]
        if stats["attachment_candidates"]
        else 0,
        "distributions": {
            k: distribution([r[k] for r in fragment_rows])
            for k in ("molecular_weight", "carbon_count", "phosphorus_count")
        },
        "nitrogen_containing": sum(r["nitrogen_count"] > 0 for r in fragment_rows),
        "sulfur_containing": sum(r["sulfur_count"] > 0 for r in fragment_rows),
        "ring_containing": sum(r["ring_containing"] for r in fragment_rows),
        "filter_flag_counts": dict(
            Counter(flag for r in fragment_rows for flag in r["filter_flags"])
        ),
        "known_head_recovery_from_queries": {
            name: any(k in fragments for k in keys) for name, keys in known_keys.items()
        },
        "representative_expanded_fragments_outside_controls": representatives,
    }
    return report


def harvest(config, config_path, output, cache_dir, stages):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    controls = read_controls(Path(config_path).parent / "known_head_controls.tsv")
    local_checks, known_keys = validate_controls(config, controls)
    log("All local chemistry/query controls passed; enabling API controls.")
    api = config.get("api", {})
    client = PubChemClient(
        cache_dir,
        requests_per_second=api.get("requests_per_second", 3),
        max_retries=api.get("max_retries", 6),
    )
    try:
        real_checks = live_controls(client, config, controls, known_keys)
        atomic_json(output / "controls.json", {"local": local_checks, "pubchem": real_checks})
        for stage in stages:
            destination = output if stage == "full" else output / stage
            destination.mkdir(parents=True, exist_ok=True)
            cap = {"stage1": 1000, "stage2": 10000, "full": api.get("full_max_records", 1000000)}[
                stage
            ]
            matches, query_results = defaultdict(set), []
            for query in config["queries"]:
                log(f"{stage}: {query['query_id']} MaxRecords={cap}")
                result, cids = run_query(client, query, cap, full=stage == "full")
                log(f"{query['query_id']}: {len(cids)} CIDs {result['status']}")
                query_results.append(result)
                for cid in cids:
                    matches[cid].add(query["query_id"])
                atomic_json(destination / "pubchem_query_summary.json", query_results)
            report = process_records(
                client, matches, config["queries"], destination, config, known_keys
            )
            overlap = {}
            for i, left in enumerate(config["queries"]):
                for right in config["queries"][i + 1 :]:
                    a, b = left["query_id"], right["query_id"]
                    overlap[f"{a}|{b}"] = sum(a in qs and b in qs for qs in matches.values())
            report.update(
                {
                    "stage": stage,
                    "query_results": query_results,
                    "unique_cids": len(matches),
                    "query_overlap": overlap,
                    "full_query_complete": stage == "full"
                    and all(r["complete"] for r in query_results),
                    "properties_complete": not report["missing_property_cids"],
                    "query_version": config["version"],
                    "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                    "rdkit_version": rdBase.rdkitVersion,
                    "client_stats": dict(client.stats),
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "local_controls": local_checks,
                    "pubchem_controls": real_checks,
                    "approved_table_generated": False,
                    "training_run": False,
                    "complete_lipid_library_generated": False,
                }
            )
            atomic_json(destination / "pubchem_polar_report.json", report)
            log(
                f"{stage} finished: {report['unique_cids']} CIDs / "
                f"{report['unique_fragments']} fragments"
            )
    finally:
        client.close()
