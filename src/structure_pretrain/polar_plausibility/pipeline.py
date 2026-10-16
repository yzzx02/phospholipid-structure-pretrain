import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from rdkit import rdBase

from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.polar_library import import_polars
from structure_pretrain.pubchem_polar.chemistry import process_smiles, read_controls

from .reactive_rules import load_rules
from .report import TIERS, stratified_examples, summarize
from .scaffold import scaffold_id
from .tiering import assign_tier, evaluate


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_tsv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def controls_check(controls_path, config, rules, rows):
    by_smiles = {row["polar_fragment_smiles"]: row for row in rows}
    result = {}
    for control in read_controls(controls_path):
        if control["kind"] != "positive":
            continue
        keys = sorted(
            {c["polar_fragment_smiles"] for c in process_smiles(control["smiles"])["candidates"]}
        )
        local = [
            evaluate({"status": "APPROVABLE", "polar_fragment_smiles": key}, config, rules)["tier"]
            for key in keys
        ]
        actual = [by_smiles[key]["tier"] if key in by_smiles else "MISSING" for key in keys]
        if any(tier not in {"CORE", "EXTENDED"} for tier in local + actual):
            raise ValueError(f"Known control lost, with no rescue allowed: {control['control_id']}")
        result[control["control_id"]] = {"local_tiers": local, "snapshot_tiers": actual}
    return result


def filter_snapshot(review_dir, output, config_path, rules_path, controls_path, log=print):
    review_dir, output = Path(review_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; preserve previous filtering runs")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
    rules = load_rules(rules_path)
    fragment_path = review_dir / "pubchem_polar_fragments.parquet"
    # Read only machine-valid records. No refetch, rescue, or reference-head similarity.
    records = pq.read_table(fragment_path, filters=[("status", "=", "APPROVABLE")]).to_pylist()
    if not records:
        raise ValueError("No APPROVABLE input records")
    records.sort(key=lambda row: row["polar_id"])
    if len({r["polar_id"] for r in records}) != len(records):
        raise ValueError("Duplicate polar_id in snapshot")
    rows = []
    for index, record in enumerate(records, 1):
        row = evaluate(record, config, rules)
        row["scaffold_id"] = (
            scaffold_id(row["polar_fragment_smiles"]) if row["tier"] in {"CORE", "EXTENDED"} else ""
        )
        rows.append(row)
        if index % 5000 == 0:
            log(f"plausibility {index}/{len(records)}", flush=True)
    controls = controls_check(controls_path, config, rules, rows)
    report = summarize(rows, controls, config["seed"])
    report["status"] = "complete"
    report["rdkit_version"] = rdBase.rdkitVersion
    report["source_snapshot"] = json.loads(
        (review_dir / "pubchem_polar_report.json").read_text(encoding="utf-8")
    ).get("properties_complete", False)
    sensitivity = {}
    for name, profile in config["profiles"].items():
        alternate = deepcopy(config)
        alternate["core"].update(
            max_molecular_weight=profile["core_mw"], max_carbon_count=profile["core_c"]
        )
        alternate_rows = []
        for row in rows:
            tier, reasons = (
                (row["tier"], row["tier_reasons"])
                if row["evaluation_error"] or row["builder_boundary_hold"]
                else assign_tier(row, row["reactive_flags"], alternate)
            )
            alternate_rows.append(dict(row, tier=tier, tier_reasons=reasons))
        alternate_controls = controls_check(controls_path, alternate, rules, alternate_rows)
        stats = summarize(alternate_rows, alternate_controls, config["seed"])
        sensitivity[name] = {
            key: stats[key]
            for key in (
                "tier_counts",
                "family_tier_counts",
                "known_controls",
                "approved_polar_count",
            )
        }
    approved = [row for row in rows if row["tier"] in {"CORE", "EXTENDED"}]
    needed_cids = {min(row["source_cids"]) for row in approved}
    originals = {}
    for batch in pq.ParquetFile(review_dir / "pubchem_polar_raw.parquet").iter_batches(
        columns=["cid", "original_smiles"], batch_size=10000
    ):
        for row in batch.to_pylist():
            if row["cid"] in needed_cids:
                originals[row["cid"]] = row["original_smiles"]
    if set(originals) != needed_cids:
        raise ValueError("Approved fragment source CID missing from original raw snapshot")
    pe_keys = {c["polar_fragment_smiles"] for c in process_smiles("OP(=O)(O)OCCN")["candidates"]}
    export_rows = []
    for row in approved:
        exact_pe = row["polar_fragment_smiles"] in pe_keys
        family = (
            "PE"
            if exact_pe
            else "known_other"
            if row["known_control_ids"]
            else "phosphonate"
            if row["polar_family"] == "phosphonate"
            else "pubchem_expanded"
        )
        export_rows.append(
            {
                "polar_id": row["polar_id"],
                "source": "PubChem",
                "source_id": json.dumps(sorted(row["source_cids"])),
                "original_smiles": originals[min(row["source_cids"])],
                "polar_fragment_smiles": row["polar_fragment_smiles"],
                "attachment_atom_map": 1,
                "polar_family": family,
                "allow_aa": True,
                "allow_oa": True,
                "allow_pa": exact_pe,
                "chemistry_family": row["polar_family"],
                "polar_plausibility_tier": row["tier"],
                "source_cids": row["source_cids"],
                "source_count": row["source_count"],
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    approved_path = output / "polar_moieties_pubchem_v0.1.tsv"
    write_tsv(approved_path, export_rows, list(export_rows[0]) if export_rows else ["polar_id"])
    rejections = []
    imported = import_polars(approved_path, rejections)
    if len(imported) != len(export_rows) or rejections:
        raise ValueError(f"Builder compatibility failed: {len(rejections)} rejections")
    report["builder_compatible_polar_count"] = len(imported)
    report["exact_pe_allow_pa_count"] = sum(row["allow_pa"] for row in export_rows)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output / "polar_plausibility_all.parquet", compression="zstd")
    for tier in TIERS:
        pq.write_table(
            table.filter(pc.equal(table["tier"], tier)),
            output / f"polar_{'rejected' if tier == 'REJECT' else tier.lower()}.parquet",
            compression="zstd",
        )
    scaffold_counts = Counter(row["scaffold_id"] for row in approved)
    dump_json(
        output / "polar_scaffold_report.json",
        {
            "scaffold_count": len(scaffold_counts),
            "fragment_count": len(approved),
            "largest_family": scaffold_counts.most_common(1),
            "scaffold_counts": dict(sorted(scaffold_counts.items())),
            "scaffold_filtering_performed": False,
        },
    )
    examples = [
        dict(row, example_set=tier)
        for tier in TIERS
        for row in stratified_examples([r for r in rows if r["tier"] == tier], 20, config["seed"])
    ]
    examples += [
        dict(row, example_set="phosphonate_approved")
        for row in stratified_examples(
            [r for r in approved if r["polar_family"] == "phosphonate"], 10, config["seed"]
        )
    ]
    examples += [
        dict(row, example_set="heterocycle_polyol_sulfur")
        for row in stratified_examples(
            [
                r
                for r in approved
                if r["heterocycle_count"] or r["contains_polyol_like"] or r["sulfur_count"]
            ],
            10,
            config["seed"],
        )
    ]
    display_fields = [
        "example_set",
        "polar_id",
        "tier",
        "polar_family",
        "polar_fragment_smiles",
        "source_cids",
        "tier_reasons",
        "reactive_flags",
    ]
    write_tsv(output / "representative_examples.tsv", examples, display_fields)
    spots = [
        dict(row, example_set=tier)
        for tier in ("CORE", "EXTENDED", "HOLD")
        for row in stratified_examples(
            [r for r in rows if r["tier"] == tier], config["spotcheck"][tier], config["seed"]
        )
    ]
    for reason in sorted(report["reject_reasons"]):
        spots += [
            dict(row, example_set=reason)
            for row in stratified_examples(
                [r for r in rows if reason in r["tier_reasons"]],
                config["spotcheck"]["per_reject_reason"],
                config["seed"],
            )
        ]
    write_tsv(output / "manual_spotcheck.tsv", spots, display_fields)
    dump_json(output / "polar_threshold_sensitivity.json", sensitivity)
    resolved = {
        **config,
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": file_hash(Path(path))}
            for name, path in {
                "fragments": fragment_path,
                "config": config_path,
                "reactive_rules": rules_path,
                "controls": controls_path,
            }.items()
        },
    }
    (output / "build_config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    report["approved_table_sha256"] = file_hash(approved_path)
    dump_json(output / "polar_plausibility_report.json", report)
    return report
