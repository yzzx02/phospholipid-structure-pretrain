import csv
import hashlib
import json
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from structure_pretrain.library_builder.canonicalize import nonisomeric_smiles, parse_molecule
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.polar_library import import_polars
from structure_pretrain.polar_plausibility.pipeline import dump_json, write_tsv
from structure_pretrain.polar_plausibility.reactive_rules import load_rules

from .chemistry import AnalogIneligible, enumerate_source, evaluate_graph, phosphate_analog

FAMILIES = ("phosphate", "phosphonate", "polyphosphate")
FIELDS = [
    "polar_id",
    "source",
    "source_id",
    "original_smiles",
    "polar_fragment_smiles",
    "attachment_atom_map",
    "polar_family",
    "allow_aa",
    "allow_oa",
    "allow_pa",
    "chemistry_family",
    "polar_plausibility_tier",
    "structure_origin",
    "source_cids",
    "source_count",
    "first_stage_status",
    "source_attachment_atom_index",
    "source_attachment_provenance",
    "parent_source_cids",
    "analog_parent_polar_ids",
    "also_generated_as_phosphate_analog",
    "fragment_formula",
    "analog_hydrogen_delta",
    "notes",
]
_CONFIG, _RULES = None, None
ANALOG_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "parent_polar_id",
            "parent_fragment_smiles",
            "polar_fragment_smiles",
            "fragment_formula",
            "parent_fragment_formula",
            "heavy_atom_delta",
            "tier",
            "merged_polar_id",
        )
    ]
    + [
        ("hydrogen_delta", pa.int64()),
        ("parent_P_atom_index", pa.int64()),
        ("parent_C_atom_index", pa.int64()),
        ("eligible", pa.bool_()),
        ("is_new_unique", pa.bool_()),
        ("reasons", pa.list_(pa.string())),
    ]
)


def read_tsv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def init_worker(plausibility_path, rules_path):
    global _CONFIG, _RULES
    _CONFIG = yaml.safe_load(Path(plausibility_path).read_text(encoding="utf-8-sig"))
    _RULES = load_rules(rules_path)


def rescue_worker(task):
    cid, smiles, targets = task
    grouped, summary = enumerate_source(cid, smiles, set(targets))
    rows = []
    for key, entry in sorted(grouped.items()):
        rows.append({**evaluate_graph(key, _CONFIG, _RULES), **entry})
    return rows, summary


def baseline_rows(path):
    rows = read_tsv(path)
    for row in rows:
        row.update(
            structure_origin="pubchem_baseline",
            first_stage_status="APPROVABLE",
            source_cids=json.loads(row["source_cids"]),
            parent_source_cids=[],
            source_attachment_atom_index=[],
            source_attachment_provenance=[],
            analog_parent_polar_ids=[],
            also_generated_as_phosphate_analog=False,
            analog_hydrogen_delta=None,
            notes="Unchanged baseline polar chemistry and permissions.",
        )
    return rows


def make_rescued(row, source_smiles):
    cids = sorted({s["cid"] for s in row["sites"]})
    sites = sorted(row["sites"], key=lambda s: (s["cid"], s["source_attachment_atom_index"]))
    return {
        "polar_id": row["polar_id"],
        "source": "PubChem",
        "source_id": json.dumps(cids),
        "original_smiles": source_smiles[cids[0]],
        "polar_fragment_smiles": row["polar_fragment_smiles"],
        "attachment_atom_map": 1,
        "polar_family": "phosphonate"
        if row["polar_family"] == "phosphonate"
        else "pubchem_expanded",
        "allow_aa": True,
        "allow_oa": True,
        "allow_pa": False,
        "chemistry_family": row["polar_family"],
        "polar_plausibility_tier": row["tier"],
        "structure_origin": "pubchem_multi_attachment_split",
        "source_cids": cids,
        "source_count": len(cids),
        "first_stage_status": "MULTIPLE_ATTACHMENT_AMBIGUITY",
        "source_attachment_atom_index": [s["source_attachment_atom_index"] for s in sites],
        "source_attachment_provenance": sites,
        "parent_source_cids": [],
        "analog_parent_polar_ids": [],
        "also_generated_as_phosphate_analog": False,
        "analog_hydrogen_delta": None,
        "notes": (
            "Site indices are zero-based RDKit atom indices from parsing original raw SMILES; "
            "paired with CID. Selected-component indices also retained."
        ),
    }


def merge_analog(rows_by_key, parent, transformed, evaluated):
    key = transformed["polar_fragment_smiles"]
    existing = rows_by_key.get(key)
    if existing is not None:
        if existing["chemistry_family"] != "phosphate":
            raise ValueError("Canonical duplicate has inconsistent phosphate family")
        existing["also_generated_as_phosphate_analog"] = True
        existing["analog_parent_polar_ids"] = sorted(
            set(existing["analog_parent_polar_ids"]) | {parent["polar_id"]}
        )
        existing["parent_source_cids"] = sorted(
            set(existing["parent_source_cids"]) | set(parent["source_cids"])
        )
        return False, existing["polar_id"]
    identifier = "PGA_" + hashlib.sha256(key.encode()).hexdigest()[:32]
    rows_by_key[key] = {
        "polar_id": identifier,
        "source": "generated_phosphate_analog",
        "source_id": "parent_polar:" + parent["polar_id"],
        "original_smiles": "",
        "polar_fragment_smiles": key,
        "attachment_atom_map": 1,
        "polar_family": "pubchem_expanded",
        "allow_aa": True,
        "allow_oa": True,
        "allow_pa": False,
        "chemistry_family": "phosphate",
        "polar_plausibility_tier": evaluated["tier"],
        "structure_origin": "generated_phosphate_analog",
        "source_cids": [],
        "source_count": 0,
        "first_stage_status": "GENERATED_NOT_PUBCHEM_RECORD",
        "source_attachment_atom_index": [],
        "source_attachment_provenance": [],
        "parent_source_cids": sorted(parent["source_cids"]),
        "analog_parent_polar_ids": [parent["polar_id"]],
        "also_generated_as_phosphate_analog": True,
        "fragment_formula": transformed["fragment_formula"],
        "analog_hydrogen_delta": transformed["hydrogen_delta"],
        "notes": (
            "Only the non-ring P-C edge was subdivided by one O. Parent CIDs are ancestry, "
            "not evidence that this generated polar or full lipid exists in PubChem."
        ),
    }
    return True, identifier


def expand_polars(
    review_dir,
    baseline_path,
    output,
    config_path,
    plausibility_path,
    rules_path,
    workers=None,
    log=print,
):
    review_dir, baseline_path, output = Path(review_dir), Path(baseline_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Expansion output must be empty; never overwrite baseline or previous runs"
        )
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
    worker_count = workers or config["workers"]
    if not 1 <= worker_count <= 4:
        raise ValueError("Use 1-4 chemistry workers")
    if config["allowed_rescue_status"] != "MULTIPLE_ATTACHMENT_AMBIGUITY":
        raise ValueError("Only MULTIPLE_ATTACHMENT_AMBIGUITY rescue is authorized")
    baseline_hash = file_hash(baseline_path)
    baseline = baseline_rows(baseline_path)
    counts = dict(Counter(r["chemistry_family"] for r in baseline))
    if baseline_hash != config["baseline_sha256"] or counts != config["baseline_family_counts"]:
        raise ValueError("Frozen baseline hash or family counts differ from the execution plan")
    plausibility = yaml.safe_load(Path(plausibility_path).read_text(encoding="utf-8-sig"))
    rules = load_rules(rules_path)
    fragments = review_dir / "pubchem_polar_fragments.parquet"
    cross = pq.read_table(fragments, columns=["polar_family", "status", "polar_id"])
    crosstab = (
        cross.group_by(["polar_family", "status"]).aggregate([("polar_id", "count")]).to_pylist()
    )
    crosstab = sorted(
        [
            {
                "chemistry_family": r["polar_family"],
                "first_stage_status": r["status"],
                "count": r["polar_id_count"],
            }
            for r in crosstab
        ],
        key=lambda r: (r["chemistry_family"], r["first_stage_status"]),
    )
    targets = pq.read_table(
        fragments, filters=[("status", "=", config["allowed_rescue_status"])]
    ).to_pylist()
    by_smiles = {r["polar_fragment_smiles"]: r for r in targets}
    by_cid = defaultdict(set)
    for row in targets:
        for cid in row["source_cids"]:
            by_cid[cid].add(row["polar_fragment_smiles"])
    originals = {}
    for batch in pq.ParquetFile(review_dir / "pubchem_polar_raw.parquet").iter_batches(
        batch_size=10000, columns=["cid", "original_smiles"]
    ):
        for row in batch.to_pylist():
            if row["cid"] in by_cid:
                originals[row["cid"]] = row["original_smiles"]
    if originals.keys() != by_cid.keys():
        raise ValueError("Rescue source CID missing from existing raw snapshot")
    output.mkdir(parents=True, exist_ok=True)
    dump_json(
        output / "baseline_manifest.json",
        {
            "sha256": baseline_hash,
            "family_counts": counts,
            "path": str(baseline_path.resolve()),
            "baseline_overwritten": False,
        },
    )
    write_tsv(
        output / "polar_family_status_crosstab.tsv",
        crosstab,
        ["chemistry_family", "first_stage_status", "count"],
    )
    evaluated, source_audit = {}, []
    tasks = [(cid, originals[cid], sorted(by_cid[cid])) for cid in sorted(by_cid)]
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(plausibility_path, rules_path),
    ) as pool:
        for index, (rows, summary) in enumerate(pool.map(rescue_worker, tasks, chunksize=16), 1):
            source_audit.append(summary)
            for row in rows:
                key = row["polar_fragment_smiles"]
                previous = evaluated.get(key)
                if previous is None:
                    row["polar_id"] = by_smiles[key]["polar_id"]
                    evaluated[key] = row
                else:
                    if (
                        previous["tier"] != row["tier"]
                        or previous["reactive_flags"] != row["reactive_flags"]
                    ):
                        raise ValueError(
                            "Identical rescued graph got different chemical classification"
                        )
                    previous["sites"].extend(row["sites"])
            if index % 1000 == 0:
                log(f"multi-attachment sources {index}/{len(tasks)}", flush=True)
    if evaluated.keys() != by_smiles.keys():
        missing = sorted(by_smiles.keys() - evaluated.keys())
        raise ValueError(
            f"Fresh enumeration did not recover {len(missing)} original multi fragments"
        )
    eval_rows = sorted(evaluated.values(), key=lambda r: r["polar_id"])
    pq.write_table(
        pa.Table.from_pylist(eval_rows), output / "rescue_evaluations.parquet", compression="zstd"
    )
    dump_json(output / "rescue_source_audit.json", source_audit)
    grouped = {}
    for row in baseline:
        key = nonisomeric_smiles(parse_molecule(row["polar_fragment_smiles"]))
        if key in grouped:
            raise ValueError("Baseline has duplicate canonical graphs")
        grouped[key] = row
    rescued, rescue_new, rescue_passed = [], Counter(), Counter()
    for row in eval_rows:
        if row["tier"] not in {"CORE", "EXTENDED"}:
            continue
        record = make_rescued(row, originals)
        family, key = record["chemistry_family"], record["polar_fragment_smiles"]
        rescue_passed[family] += 1
        is_new = key not in grouped
        if is_new:
            grouped[key] = record
            rescue_new[family] += 1
        rescued.append(dict(record, is_new_unique=is_new, merged_polar_id=grouped[key]["polar_id"]))
    write_tsv(
        output / "rescued_multi_attachment.tsv",
        rescued,
        FIELDS + ["is_new_unique", "merged_polar_id"],
    )
    after_rescue = Counter(r["chemistry_family"] for r in grouped.values())
    analog_needed = after_rescue["phosphate"] < after_rescue["phosphonate"]
    analog_rows, analog_counts, analog_tiers = [], Counter(), Counter()
    if analog_needed:
        parents = sorted(
            [r for r in grouped.values() if r["chemistry_family"] == "phosphonate"],
            key=lambda r: r["polar_id"],
        )
        for index, parent in enumerate(parents, 1):
            audit = {
                "parent_polar_id": parent["polar_id"],
                "parent_fragment_smiles": parent["polar_fragment_smiles"],
            }
            try:
                transformed = phosphate_analog(parent["polar_fragment_smiles"])
                analog_counts["eligible_phosphonate"] += 1
                evaluated_analog = evaluate_graph(
                    transformed["polar_fragment_smiles"], plausibility, rules
                )
                analog_tiers[evaluated_analog["tier"]] += 1
                audit.update(
                    transformed,
                    tier=evaluated_analog["tier"],
                    reasons=evaluated_analog["tier_reasons"],
                    eligible=True,
                )
                audit["heavy_atom_delta"] = json.dumps(
                    transformed["heavy_atom_delta"], sort_keys=True
                )
                if evaluated_analog["tier"] in {"CORE", "EXTENDED"}:
                    analog_counts["passed_generations"] += 1
                    previous_origin = grouped.get(transformed["polar_fragment_smiles"], {}).get(
                        "structure_origin"
                    )
                    is_new, identifier = merge_analog(
                        grouped, parent, transformed, evaluated_analog
                    )
                    analog_counts[
                        "new_unique_phosphate" if is_new else "duplicate_existing_phosphate"
                    ] += 1
                    audit.update(is_new_unique=is_new, merged_polar_id=identifier)
                    if not is_new:
                        analog_counts[
                            "duplicate_other_generated_analog"
                            if previous_origin == "generated_phosphate_analog"
                            else "duplicate_baseline_or_rescued_phosphate"
                        ] += 1
            except AnalogIneligible as exc:
                audit.update(eligible=False, tier="INELIGIBLE", reasons=[str(exc)])
                analog_counts[str(exc)] += 1
            except (ValueError, RuntimeError) as exc:
                audit.update(
                    eligible=False, tier="REJECT", reasons=["ANALOG_TRANSFORM_ERROR", str(exc)]
                )
                analog_counts["ANALOG_TRANSFORM_ERROR"] += 1
            analog_rows.append(audit)
            if index % 5000 == 0:
                log(f"phosphate analogs {index}/{len(parents)}", flush=True)
    pq.write_table(
        pa.Table.from_pylist(analog_rows, schema=ANALOG_SCHEMA),
        output / "phosphate_analog_audit.parquet",
        compression="zstd",
    )
    expanded = sorted(grouped.values(), key=lambda r: r["polar_id"])
    export_path = output / "polar_moieties_expanded_v0.2.tsv"
    write_tsv(export_path, expanded, FIELDS)
    errors = []
    imported = import_polars(export_path, errors)
    if errors or len(imported) != len(expanded):
        raise ValueError(f"Expanded builder compatibility failed: {len(errors)} rejections")
    if file_hash(baseline_path) != baseline_hash:
        raise ValueError("Baseline changed during expansion")
    if not {r["polar_id"] for r in baseline}.issubset(r["polar_id"] for r in expanded):
        raise ValueError("Baseline head lost during expansion")
    report = {
        "status": "complete",
        "baseline_family_counts": counts,
        "baseline_sha256": baseline_hash,
        "first_stage_family_status": crosstab,
        "review_fragment_count": cross.num_rows,
        "multi_input_family_counts": dict(Counter(r["polar_family"] for r in targets)),
        "multi_source_cid_count": len(tasks),
        "multi_reenumerated_site_count": sum(r["enumerated_sites"] for r in source_audit),
        "rescue_tier_counts": dict(Counter(r["tier"] for r in eval_rows)),
        "rescued_passed_by_family": dict(rescue_passed),
        "rescued_new_unique_by_family": dict(rescue_new),
        "after_rescue_family_counts": dict(after_rescue),
        "analog_stage_needed": analog_needed,
        "analog_counts": dict(analog_counts),
        "analog_tier_counts": dict(analog_tiers),
        "expanded_family_counts": dict(Counter(r["chemistry_family"] for r in expanded)),
        "expanded_origin_counts": dict(Counter(r["structure_origin"] for r in expanded)),
        "approved_polar_count": len(expanded),
        "approved_table_sha256": file_hash(export_path),
        "baseline_retained_count": len(baseline),
        "empirical_polar_count": sum(
            r["structure_origin"] != "generated_phosphate_analog" for r in expanded
        ),
        "phosphonate_heads_removed_for_balance": 0,
        "charge_or_filter_review_rescued": 0,
        "network_requests": 0,
        "training_performed": False,
        "full_lipid_library_generated": False,
    }
    dump_json(output / "polar_expansion_report.json", report)
    resolved = {
        **config,
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": file_hash(Path(path))}
            for name, path in {
                "baseline": baseline_path,
                "fragments": fragments,
                "config": config_path,
                "plausibility": plausibility_path,
                "reactive_rules": rules_path,
            }.items()
        },
    }
    (output / "build_config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True), encoding="utf-8"
    )
    return report
