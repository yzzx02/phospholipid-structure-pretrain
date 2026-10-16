"""Bounded-memory, resumable candidate assembly and global graph deduplication."""

import hashlib
import json
import multiprocessing
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from statistics import median

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from rdkit import DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.polar_plausibility.pipeline import dump_json

from .assembly import assemble
from .canonicalize import parse_molecule
from .chain_combinations import make_combination, permitted
from .chain_library import import_chains
from .coverage import chain_coverage
from .family_sampling import allocate_family_budget, planned_extras
from .full_library_v02 import POLARS_V02, STRUCTURES_V02, load_config_v02, preflight_v02
from .parquet_io import CHAIN_SCHEMA, COMBINATION_SCHEMA, REJECTION_SCHEMA, write_rows
from .pipeline import file_hash
from .sampling import StratifiedSampler
from .schemas import ELEMENTS, LibraryValidationError

FINAL_SCHEMA = pa.schema(
    list(STRUCTURES_V02)
    + [
        pa.field("polar_fragment_smiles", pa.string()),
        pa.field("morgan_fp_2048", pa.binary(256)),
        pa.field("is_mandatory_pair", pa.bool_()),
    ],
    metadata={b"morgan": b"radius=2;fpSize=2048;includeChirality=false;RDKit binary text"},
)
_CHAINS = _CONFIG = _GENERATOR = None


@lru_cache(maxsize=4096)
def cached_fragment(smiles):
    # assemble never mutates its source fragments; CombineMols returns a new graph.
    return parse_molecule(smiles)


def validate_formula(row):
    formula = row["neutral_formula"]
    matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if "".join(a + b for a, b in matches) != formula:
        raise LibraryValidationError(
            "formula_invalid", "Neutral formula contains unsupported syntax"
        )
    parsed = {e: int(n or 1) for e, n in matches}
    if set(parsed) - set(ELEMENTS) or any(
        parsed.get(e, 0) != row[f"neutral_formula_{e}"] for e in ELEMENTS
    ):
        raise LibraryValidationError("formula_invalid", "Formula and graph element counts disagree")


def init_assembly(chains_path, config_path):
    global _CHAINS, _CONFIG, _GENERATOR
    errors = []
    _CHAINS = import_chains(chains_path, errors)
    if errors:
        raise ValueError("Worker chain input validation failed")
    _CONFIG, _ = load_config_v02(config_path)
    _GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )


def assemble_shard(task):
    number, work_dir, entries = task
    path = Path(work_dir) / f"candidates_{number:05d}.parquet"
    stats_path = path.with_suffix(".json")
    signature = hashlib.sha256(repr(entries).encode()).hexdigest()
    if path.exists() and stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats["job_signature"] != signature or file_hash(path) != stats["sha256"]:
            raise ValueError("Resume shard differs from the approved schedule")
        return stats
    start = time.monotonic()
    stats = {
        "job": number,
        "job_signature": signature,
        "planned": 0,
        "assembled": 0,
        "reasons": {},
        "mandatory_failures": [],
        "seconds": 0,
    }
    errors, reasons = [], Counter()

    def rows():
        for polar, details, combos in entries:
            for label, mandatory in combos:
                stats["planned"] += 1
                try:
                    combo = make_combination(tuple(label.split("_")), _CHAINS)
                    if not permitted(combo, polar, _CONFIG):
                        raise LibraryValidationError(
                            "chain_policy", "Unpermitted head/chain pairing"
                        )
                    row = assemble(_CHAINS, combo, polar, fragment_parser=cached_fragment)
                    validate_formula(row)
                    row.update(details)
                    row["polar_fragment_smiles"] = polar.polar_fragment_smiles
                    row["structure_id"] = (
                        "pl_" + hashlib.sha256(row["canonical_smiles"].encode()).hexdigest()
                    )
                    row["is_anchor_combination"] = label in _CONFIG.anchor_combinations
                    row["is_mandatory_pair"] = mandatory
                    fp = _GENERATOR.GetFingerprint(parse_molecule(row["canonical_smiles"]))
                    row["morgan_fp_2048"] = DataStructs.BitVectToBinaryText(fp)
                    stats["assembled"] += 1
                    yield row
                except LibraryValidationError as exc:
                    reasons[exc.reason] += 1
                    identifier = f"{polar.polar_id}/{label}"
                    errors.append(
                        {
                            "stage": "assembly",
                            "record_id": identifier,
                            "reason": exc.reason,
                            "message": str(exc),
                        }
                    )
                    if mandatory:
                        stats["mandatory_failures"].append(identifier)

    partial = path.with_suffix(".partial")
    with rdBase.BlockLogs():
        write_rows(partial, rows(), FINAL_SCHEMA)
    partial.replace(path)
    dump_json(path.with_suffix(".rejections.json"), errors)
    stats.update(reasons=dict(reasons), seconds=time.monotonic() - start, sha256=file_hash(path))
    dump_json(stats_path, stats)
    return stats


def scheduled_jobs(polars, details, selected, bank, config, weights, work, digest, heads_per_job):
    budget = allocate_family_budget(polars, details, selected, config, weights)
    sampler = StratifiedSampler(bank, config)
    chunk, number = [], 0
    for polar in polars:
        combos = [(label, True) for label in sorted(selected[polar.polar_id])]
        combos.extend(
            (c.combination_id, False)
            for c in planned_extras(polar, selected[polar.polar_id], budget, sampler)
        )
        for label, _ in combos:
            digest.update(f"{polar.polar_id}\t{label}\n".encode())
        chunk.append((polar, details[polar.polar_id], combos))
        if len(chunk) == heads_per_job:
            yield number, str(work), chunk
            number += 1
            chunk = []
    if chunk:
        yield number, str(work), chunk


def combine_shards(output, work, stats, chains, polars, selected, config, dry):
    connection = sqlite3.connect(work / "canonical_seen.sqlite")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA cache_size=-32768")
    # Export may be retried after interruption; regenerate only the small key index.
    connection.execute("DROP TABLE IF EXISTS seen")
    connection.execute("CREATE TABLE seen (smiles TEXT PRIMARY KEY) WITHOUT ROWID")
    per_polar = Counter({p.polar_id: 0 for p in polars})
    tokens, families, linkages, tiers, origins = (Counter() for _ in range(5))
    special, anchors = Counter(), Counter({p.polar_id: 0 for p in polars})
    totals = Counter()
    special_rows, duplicate_rows, rejections = [], [], []
    for s in stats:
        rejections.extend(
            json.loads(
                (work / f"candidates_{s['job']:05d}.rejections.json").read_text(encoding="utf-8")
            )
        )

    def rows():
        for stat in sorted(stats, key=lambda s: s["job"]):
            path = work / f"candidates_{stat['job']:05d}.parquet"
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2000):
                for row in batch.to_pylist():
                    added = connection.execute(
                        "INSERT OR IGNORE INTO seen VALUES (?)", (row["canonical_smiles"],)
                    ).rowcount
                    if not added:
                        totals["duplicates"] += 1
                        duplicate_rows.append(
                            {
                                k: row[k]
                                for k in (
                                    "structure_id",
                                    "polar_id",
                                    "combination_id",
                                    "structure_origin",
                                )
                            }
                        )
                        continue
                    totals["unique"] += 1
                    per_polar[row["polar_id"]] += 1
                    tokens.update(set(row["chain_tokens"]))
                    families[row["chemistry_family"]] += 1
                    linkages["lyso" if row["is_lyso"] else row["linkage_family"]] += 1
                    tiers[row["polar_plausibility_tier"]] += 1
                    origins[row["structure_origin"]] += 1
                    anchors[row["polar_id"]] += row["is_anchor_combination"]
                    if row["is_special_case"]:
                        special[row["polar_id"]] += 1
                        special_rows.append(row)
                    yield row
            connection.commit()
            dump_json(
                output / "progress.json",
                {
                    "stage": "global_dedup_export",
                    "job": stat["job"],
                    "unique_written": totals["unique"],
                    "duplicates": totals["duplicates"],
                },
            )

    try:
        write_rows(output / "structures.parquet", rows(), FINAL_SCHEMA)
    finally:
        connection.close()
    write_rows(output / "special_cases.parquet", special_rows, FINAL_SCHEMA)
    write_rows(output / "rejections.parquet", rejections, REJECTION_SCHEMA)
    duplicate_schema = pa.schema(
        [
            (k, pa.string())
            for k in ("structure_id", "polar_id", "combination_id", "structure_origin")
        ]
    )
    write_rows(output / "canonical_duplicate_provenance.parquet", duplicate_rows, duplicate_schema)
    chain_report = chain_coverage(chains, tokens)
    dump_json(output / "chain_coverage_report.json", chain_report)
    polar_rows = [
        {"polar_id": p, "final_full_structure_count": n} for p, n in sorted(per_polar.items())
    ]
    pq.write_table(
        pa.Table.from_pylist(polar_rows), output / "polar_coverage.parquet", compression="zstd"
    )
    reasons = Counter(r["reason"] for r in rejections)
    planned = sum(s["planned"] for s in stats)
    assembled = sum(s["assembled"] for s in stats)
    audit_passed = (
        chain_report["status"] == "passed"
        and min(per_polar.values()) >= 3
        and min(anchors.values()) >= 3
        and len(special) == len(dry["special_head_background_counts"])
        and assembled + len(rejections) == planned
        and totals["unique"] + totals["duplicates"] == assembled
        and planned == dry["estimated_full_structures"] <= config.max_structures
    )
    report = {
        "status": "complete" if audit_passed else "audit_failed",
        "schema_version": config.schema_version,
        "planned_head_chain_pairs": planned,
        "successfully_assembled": assembled,
        "sanitize_rejected": sum(
            reasons[r] for r in ("invalid_valence", "sanitize_failed", "invalid_smiles")
        ),
        "formula_rejected": reasons["formula_invalid"],
        "rejection_reasons": dict(reasons),
        "total_rejected": len(rejections),
        "canonical_duplicates": totals["duplicates"],
        "final_unique_full_structures": totals["unique"],
        "full_by_chemistry_family": dict(families),
        "AA": linkages["AA"],
        "OA": linkages["OA"],
        "PE_PA": linkages["PA"],
        "lyso": linkages["lyso"],
        "by_plausibility_tier": dict(tiers),
        "by_structure_origin": dict(origins),
        "expected_polar_count": len(polars),
        "covered_polar_count": sum(n > 0 for n in per_polar.values()),
        "structures_per_polar_min": min(per_polar.values()),
        "structures_per_polar_median": median(per_polar.values()),
        "structures_per_polar_max": max(per_polar.values()),
        "common_anchor_coverage_min": min(anchors.values()),
        "special_head_background_counts": dict(special),
        "special_full_structure_count": sum(special.values()),
        "all_chain_coverage_passed": chain_report["status"] == "passed",
        "within_2000000": planned <= 2_000_000,
        "fingerprint": {
            "radius": 2,
            "bits": 2048,
            "includeChirality": False,
            "rdkit_version": rdBase.rdkitVersion,
        },
        "full_lipid_library_generated": True,
        "training_performed": False,
        "dataset_split_created": False,
        "provenance_used_for_labels_or_weights": False,
        "no_automatic_refill": True,
    }
    dump_json(output / "library_report.json", report)
    if not audit_passed:
        raise ValueError("Final coverage/count audit failed; inspect library_report.json")
    return report


def build_production(
    preparation,
    chains_path,
    polar_path,
    report_path,
    special_dir,
    output,
    work,
    workers=4,
    heads_per_job=800,
):
    preparation, output, work = map(Path, (preparation, output, work))
    manifest = json.loads((preparation / "input_hashes.json").read_text(encoding="utf-8"))
    for entry in manifest.values():
        if file_hash(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Approved preparation inputs changed")
    config_path = preparation / "library_build_2m_final.yaml"
    dry = json.loads((preparation / "family_dry_run_v02.json").read_text(encoding="utf-8"))
    coverage = json.loads(
        (preparation / "chain_token_coverage_dry_run.json").read_text(encoding="utf-8")
    )
    if coverage["status"] != "passed" or dry["estimated_full_structures"] != 2_000_000:
        raise ValueError("Formal build blocked by preparation coverage/budget")
    config, options = load_config_v02(config_path)
    _, (chains, polars, details, selected, bank) = preflight_v02(
        config, options, chains_path, polar_path, report_path, special_dir
    )
    for key, path in {"chains": chains_path, "polar": polar_path, "report": report_path}.items():
        if file_hash(Path(path)) != manifest[key]["sha256"]:
            raise ValueError("Formal CLI input differs from preparation")
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if (output / "library_report.json").exists():
        existing = json.loads((output / "library_report.json").read_text(encoding="utf-8"))
        if existing["status"] == "complete":
            raise ValueError("A completed formal library already exists; preserve it")
    dump_json(output / "input_hashes.json", manifest)
    (output / "build_config_resolved.yaml").write_text(
        yaml.safe_dump({**asdict(config), **asdict(options)}, sort_keys=False), encoding="utf-8"
    )
    write_rows(output / "chain_library.parquet", (c.row() for c in chains.values()), CHAIN_SCHEMA)
    write_rows(
        output / "polar_library.parquet",
        ({**p.row(), **details[p.polar_id]} for p in polars),
        POLARS_V02,
    )
    write_rows(output / "chain_combinations.parquet", (c.row() for c in bank), COMBINATION_SCHEMA)
    digest = hashlib.sha256()
    jobs = iter(
        scheduled_jobs(
            polars,
            details,
            selected,
            bank,
            config,
            options.family_sampling_weight,
            work,
            digest,
            heads_per_job,
        )
    )
    results, pending = [], []
    start = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_assembly,
        initargs=(str(chains_path), str(config_path)),
    ) as pool:
        exhausted = False
        while pending or not exhausted:
            while len(pending) < workers * 2 and not exhausted:
                task = next(jobs, None)
                if task is None:
                    exhausted = True
                else:
                    pending.append(pool.submit(assemble_shard, task))
            if not pending:
                break
            stat = pending.pop(0).result()
            results.append(stat)
            progress = {
                "stage": "assembly",
                "finished_jobs": len(results),
                "planned_pairs_finished": sum(s["planned"] for s in results),
                "successfully_assembled": sum(s["assembled"] for s in results),
                "seconds": time.monotonic() - start,
            }
            dump_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    if digest.hexdigest() != dry["schedule_sha256"]:
        raise ValueError("Formal schedule differs from the approved dry-run")
    dump_json(output / "assembly_shards_manifest.json", results)
    report = combine_shards(output, work, results, chains, polars, selected, config, dry)
    dump_json(
        output / "progress.json",
        {
            "stage": "complete",
            "seconds": time.monotonic() - start,
            "final_unique": report["final_unique_full_structures"],
        },
    )
    return report
