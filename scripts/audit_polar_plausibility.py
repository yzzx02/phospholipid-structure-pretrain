"""Cross-check the emitted filtering package without fetching or training."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_plausibility.pipeline import dump_json


def audit(review_dir, output):
    output = Path(output)
    report = json.loads((output / "polar_plausibility_report.json").read_text(encoding="utf-8"))
    original = pq.read_table(
        Path(review_dir) / "pubchem_polar_fragments.parquet",
        columns=["polar_id", "polar_fragment_smiles", "source_cids", "status"],
        filters=[("status", "=", "APPROVABLE")],
    ).to_pylist()
    original = {r["polar_id"]: r for r in original}
    table = pq.read_table(output / "polar_plausibility_all.parquet")
    rows = table.to_pylist()
    actual = {r["polar_id"]: r for r in rows}
    assert len(rows) == len(actual) == len(original) == report["input_approvable_count"]
    assert actual.keys() == original.keys()
    for identifier, row in actual.items():
        assert row["polar_fragment_smiles"] == original[identifier]["polar_fragment_smiles"]
        assert row["source_cids"] == original[identifier]["source_cids"]
        assert row["status"] == "APPROVABLE"
    counts = Counter(r["tier"] for r in rows)
    assert dict(counts) == report["tier_counts"]
    for tier in ("CORE", "EXTENDED", "HOLD", "REJECT"):
        name = "rejected" if tier == "REJECT" else tier.lower()
        saved = pq.read_table(output / f"polar_{name}.parquet")
        assert saved.equals(table.filter(pc.equal(table["tier"], tier)))
    approved = {r["polar_id"] for r in rows if r["tier"] in {"CORE", "EXTENDED"}}
    approved_path = output / "polar_moieties_pubchem_v0.1.tsv"
    with approved_path.open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream, delimiter="\t"))
    assert {r["polar_id"] for r in exported} == approved
    assert len(exported) == report["approved_polar_count"]
    assert file_hash(approved_path) == report["approved_table_sha256"]
    for row in exported:
        origin = actual[row["polar_id"]]
        assert row["polar_fragment_smiles"] == origin["polar_fragment_smiles"]
        assert json.loads(row["source_id"]) == sorted(origin["source_cids"])
        assert row["chemistry_family"] == origin["polar_family"]
        assert row["polar_plausibility_tier"] == origin["tier"]
        assert row["allow_aa"] == row["allow_oa"] == "True"
    pe_rows = [r for r in exported if r["allow_pa"] == "True"]
    assert len(pe_rows) == 1 and pe_rows[0]["polar_family"] == "PE"
    scaffold = json.loads((output / "polar_scaffold_report.json").read_text())
    assert sum(scaffold["scaffold_counts"].values()) == len(approved)
    assert scaffold["scaffold_count"] == len(scaffold["scaffold_counts"])
    assert all(r["tier"] == "REJECT" for r in rows if r["reactive_flags"])
    assert sum(bool(r["reactive_flags"]) for r in rows) == report["reactive_hard_reject_count"]
    assert report["rule_exception_count"] == sum(bool(r["evaluation_error"]) for r in rows) == 0
    assert sum(r["builder_boundary_hold"] for r in rows) == report["builder_boundary_hold_count"]
    assert all(r["tier"] == "HOLD" for r in rows if r["builder_boundary_hold"])
    assert all(c["snapshot_tiers"] == ["CORE"] for c in report["known_controls"].values())
    assert report["training_performed"] is report["full_lipid_library_generated"] is False
    result = {
        "status": "passed",
        "rows_checked": len(rows),
        "approved_rows_checked": len(exported),
        "all_original_fragment_ids_smiles_and_cids_preserved": True,
        "four_way_partition_exact": True,
        "tsv_hash_and_provenance_verified": True,
        "exact_pe_only_pa_verified": True,
        "builder_boundary_held_not_rejected": True,
        "training_performed": False,
        "full_lipid_library_generated": False,
    }
    dump_json(output / "verification_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", default="data/private/pubchem_polar_review")
    parser.add_argument("--output", default="data/private/pubchem_polar_plausibility_v0.1")
    args = parser.parse_args()
    print(json.dumps(audit(args.review_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
