import csv
import hashlib
import itertools
import json
import multiprocessing
import os
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from .cache import ChemistryCache
from .chemistry import (
    Reject,
    acidic_sites,
    process_smiles,
    read_controls,
    select_component,
    validate_controls,
)
from .client import PubChemClient, atomic_json, run_query
from .fragment_store import FragmentStore


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


def validate_remote_queries(client, config, live_checks):
    """RDKit SMARTS matches do not prove PubChem's explicit-H search semantics."""
    results = {}
    for query in config["queries"]:
        cids = set(client.query(query["smarts"], 1000))
        expected = {
            row["cid"]
            for name in query["positive_controls"]
            for row in live_checks[name]["records"]
        }
        missing = sorted(expected - cids)
        results[query["query_id"]] = {
            "expected_cids": sorted(expected),
            "missing_cids": missing,
            "passed": not missing,
        }
        if missing:
            raise RuntimeError(
                f"Server-side positive CID controls not demonstrated for "
                f"{query['query_id']}: {missing}; stop before large queries"
            )
        log(f"server-side CID controls {query['query_id']}: passed")
    return results


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


def property_batches(client, cids, batch_size, workers=1):
    """Small ordered prefetch queue; all senders share one client rate limiter."""
    if not cids:
        return
    if not 1 <= workers <= 2:
        raise ValueError("Property workers must be 1 or 2")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pubchem-properties") as pool:
        pending, starts = deque(), iter(range(0, len(cids), batch_size))

        def submit_next():
            start = next(starts, None)
            if start is not None:
                batch = cids[start : start + batch_size]
                pending.append((start, batch, pool.submit(client.get_properties, batch)))

        for _ in range(workers):
            submit_next()
        while pending:
            start, batch, future = pending.popleft()
            try:
                properties, error = future.result(), None
            except Exception as exc:
                properties, error = [], exc
            submit_next()
            yield start, batch, properties, error


def merge_candidate(fragments, candidate, cid, query_ids, known_keys):
    if isinstance(fragments, FragmentStore):
        fragments.merge(candidate, cid, query_ids, known_keys)
        return
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


def representative_sort_key(row):
    """Prioritize interpretable display examples; never filter the actual review table."""
    mol = Chem.MolFromSmiles(row["polar_fragment_smiles"])
    unusual_oxygen_bond = any(
        bond.GetBondType() == Chem.BondType.SINGLE
        and any(
            atom.GetAtomicNum() == 8 and bond.GetOtherAtom(atom).GetAtomicNum() not in (1, 6, 15)
            for atom in (bond.GetBeginAtom(), bond.GetEndAtom())
        )
        for bond in mol.GetBonds()
    )
    small_ring = any(len(ring) < 5 for ring in mol.GetRingInfo().AtomRings())
    penalty = int(row["carbon_count"] == 0) + int(unusual_oxygen_bond) + int(small_ring)
    return penalty, row["heavy_atoms"], row["polar_id"]


def analyze_bundle(smiles, limits):
    bundle = {"component": None, "candidates": [], "rejection": None}
    try:
        selected, removed = select_component(smiles)
        bundle["component"] = {
            "selected_p_component": Chem.MolToSmiles(selected),
            "removed_counterions": removed,
            "source_component_charge": Chem.GetFormalCharge(selected),
            "attachment_site_count": len(acidic_sites(selected)),
            "distinct_attachment_fragments": None,
        }
        processed = process_smiles(smiles, limits, selected=(selected, removed))
        bundle["component"]["distinct_attachment_fragments"] = processed[
            "distinct_attachment_fragments"
        ]
        bundle["candidates"] = processed["candidates"]
    except Reject as exc:
        bundle["rejection"] = {"reason": exc.reason, "detail": str(exc)}
    return bundle


def batch_bundles(rows, limits, cache=None, executor=None):
    prepared = [
        [row, cache.get(row["cid"], row["original_smiles"]) if cache else None, True]
        for row in rows
    ]
    missing = [i for i, item in enumerate(prepared) if item[1] is None]
    smiles = [prepared[i][0]["original_smiles"] for i in missing]
    results = (
        executor.map(analyze_bundle, smiles, itertools.repeat(limits), chunksize=16)
        if executor
        else (analyze_bundle(smi, limits) for smi in smiles)
    )
    for i, bundle in zip(missing, results):
        row = prepared[i][0]
        prepared[i][1], prepared[i][2] = bundle, False
        if cache:
            cache.put(row["cid"], row["original_smiles"], bundle)
    return prepared


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
    fragments = (
        FragmentStore(client.cache_dir, FRAGMENT_SCHEMA.names, [q["query_id"] for q in queries])
        if hasattr(client, "cache_dir")
        else {}
    )
    reasons, stats = Counter(), Counter()
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
    chemistry_cache = (
        ChemistryCache(client.cache_dir, config.get("limits", {}))
        if hasattr(client, "cache_dir")
        else None
    )
    workers = min(max(1, int(config.get("chemistry_workers", 1))), os.cpu_count() or 1)
    executor = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1 and cids
        else None
    )
    property_workers = config.get("api", {}).get("property_workers", 1)
    for start, batch, properties, error in property_batches(
        client, cids, batch_size, workers=property_workers
    ):
        if error is not None:
            # Preserve all unprocessed CIDs; do not misclassify network failures as chemistry.
            missing_cids.extend(batch)
            log(f"properties batch failed ({len(batch)} CIDs): {error}")
            continue
        returned_cids = {prop["CID"] for prop in properties}
        missing_cids.extend(cid for cid in batch if cid not in returned_cids)
        raw_rows = [
            normalize_raw(prop, matches[prop["CID"]], {families[q] for q in matches[prop["CID"]]})
            for prop in properties
        ]
        for row, bundle, from_cache in batch_bundles(
            raw_rows, config.get("limits"), chemistry_cache, executor
        ):
            cid = row["cid"]
            sinks["polar_raw"].add(row)
            stats["properties_downloaded"] += 1
            stats["chemistry_cache_hits" if from_cache else "chemistry_computed"] += 1
            component = bundle["component"]
            if component is not None:
                sinks["p_components"].add(
                    {"cid": cid, "original_smiles": row["original_smiles"], **component}
                )
                stats["single_p_component_records"] += 1
                stats["counterion_records"] += bool(component["removed_counterions"])
                stats["counterions_removed"] += len(component["removed_counterions"])
            if bundle["rejection"] is None:
                stats["attachment_eligible_cids"] += 1
                stats["attachment_candidates"] += len(bundle["candidates"])
                stats["unique_cid_fragment_pairs"] += component["distinct_attachment_fragments"]
                for candidate in bundle["candidates"]:
                    sinks["attachment_candidates"].add(
                        {"cid": cid, **candidate, "matched_query_ids": sorted(matches[cid])}
                    )
                    merge_candidate(fragments, candidate, cid, matches[cid], known_keys)
            else:
                rejection = bundle["rejection"]
                reasons[rejection["reason"]] += 1
                sinks["polar_rejections"].add(
                    {
                        "cid": cid,
                        "original_smiles": row["original_smiles"],
                        **rejection,
                        "matched_query_ids": sorted(matches[cid]),
                    }
                )
        if chemistry_cache:
            chemistry_cache.flush()
        if isinstance(fragments, FragmentStore):
            fragments.flush()
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
    chemistry_policy_hash = chemistry_cache.policy_hash if chemistry_cache else None
    if executor:
        executor.shutdown()
    if chemistry_cache:
        chemistry_cache.close()
    for sink in sinks.values():
        sink.close()
    family_counts, status_counts, filter_counts, descriptors = Counter(), Counter(), Counter(), {}
    descriptors = {key: [] for key in ("molecular_weight", "carbon_count", "phosphorus_count")}
    charge_counts = dict.fromkeys(
        (
            "nonzero_formal_charge",
            "neutral_multiple_charge_assignments",
            "also_soft_limit_review",
            "also_long_chain_review",
        ),
        0,
    )
    other_counts, buckets = Counter(), {}
    fragment_sink = ParquetSink(output / "pubchem_polar_fragments.parquet", FRAGMENT_SCHEMA)
    fragment_rows = (
        fragments.iter_rows()
        if isinstance(fragments, FragmentStore)
        else iter(sorted(fragments.values(), key=lambda r: r["polar_id"]))
    )
    exported = 0
    with (output / "pubchem_polar_review.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAGMENT_SCHEMA.names, delimiter="\t")
        writer.writeheader()
        for row in fragment_rows:
            for name in ("source_cids", "matched_query_ids", "review_flags", "filter_flags"):
                row[name] = sorted(row[name])
            row["source_count"] = len(row["source_cids"])
            fragment_sink.add(row)
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                    for k, v in row.items()
                }
            )
            family_counts[row["polar_family"]] += 1
            status_counts[row["status"]] += 1
            filter_counts.update(row["filter_flags"])
            for key in descriptors:
                descriptors[key].append(row[key])
            for key in ("nitrogen", "sulfur"):
                other_counts[key + "_containing"] += row[key + "_count"] > 0
            other_counts["ring_containing"] += row["ring_containing"]
            if row["status"] == "CHARGE_AMBIGUOUS":
                charge_counts["nonzero_formal_charge"] += row["formal_charge"] != 0
                charge_counts["neutral_multiple_charge_assignments"] += row["formal_charge"] == 0
                charge_counts["also_soft_limit_review"] += bool(row["filter_flags"])
                charge_counts["also_long_chain_review"] += (
                    "long_aliphatic_carbon_run" in row["filter_flags"]
                )
            if row["status"] == "APPROVABLE" and not row["known_control_ids"]:
                bucket = buckets.setdefault(row["polar_family"], [])
                bucket.append((representative_sort_key(row), row))
                bucket.sort(key=lambda item: item[0])
                if len(bucket) > 20:
                    bucket.pop()
            exported += 1
    fragment_sink.close()
    if exported != len(fragments):
        raise ValueError("Fragment store/export count mismatch")
    representatives = []
    while len(representatives) < 20 and any(buckets.values()):
        for family in family_counts:
            bucket = buckets.get(family, [])
            if bucket and len(representatives) < 20:
                representatives.append(bucket.pop(0)[1])
    pairs = stats["unique_cid_fragment_pairs"]
    report = {
        **dict(stats),
        "chemistry_policy_sha256": chemistry_policy_hash,
        "chemistry_workers": workers,
        "property_workers": property_workers,
        "unique_fragments": exported,
        "fragment_aggregation": "sqlite-ssd" if isinstance(fragments, FragmentStore) else "memory",
        "family_counts": dict(family_counts),
        "status_counts": dict(status_counts),
        "charge_review_breakdown": charge_counts,
        "rejections": dict(reasons),
        "missing_property_cids": missing_cids,
        "cid_fragment_pair_dedup_rate": 1 - exported / pairs if pairs else 0,
        "attachment_candidate_dedup_rate": 1 - exported / stats["attachment_candidates"]
        if stats["attachment_candidates"]
        else 0,
        "distributions": {key: distribution(values) for key, values in descriptors.items()},
        **{
            key: other_counts[key]
            for key in ("nitrogen_containing", "sulfur_containing", "ring_containing")
        },
        "filter_flag_counts": dict(filter_counts),
        "known_head_recovery_from_queries": {
            name: any(k in fragments for k in keys) for name, keys in known_keys.items()
        },
        "representative_expanded_fragments_outside_controls": representatives,
    }
    if isinstance(fragments, FragmentStore):
        fragments.close()
    return report


def harvest(config, config_path, output, cache_dir, stages, cached_only=False):
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
        cached_only=cached_only,
    )
    try:
        real_checks = live_controls(client, config, controls, known_keys)
        remote_checks = validate_remote_queries(client, config, real_checks)
        atomic_json(
            output / "controls.json",
            {"local": local_checks, "pubchem": real_checks, "server_side_queries": remote_checks},
        )
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
                    "property_acquisition_mode": "cached-only" if cached_only else "online",
                    "network_suspension": client.network_suspension,
                    "run_status": (
                        "PARTIAL"
                        if report["missing_property_cids"]
                        or not all(r["complete"] for r in query_results)
                        else "COMPLETE"
                    ),
                    "query_version": config["version"],
                    "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                    "rdkit_version": rdBase.rdkitVersion,
                    "client_stats": dict(client.stats),
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "local_controls": local_checks,
                    "pubchem_controls": real_checks,
                    "pubchem_query_controls": remote_checks,
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
