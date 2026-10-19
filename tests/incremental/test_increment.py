import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import build_combination_bank
from structure_pretrain.library_builder.chain_library import import_chains
from structure_pretrain.library_builder.coverage import chain_coverage
from structure_pretrain.library_builder.full_library_v02 import metadata
from structure_pretrain.library_builder.incremental import (
    accept_unique,
    append_baseline,
    build_increment,
    freeze_files,
    low_percentile,
    verify_frozen,
    verify_prefix,
)
from structure_pretrain.library_builder.parquet_io import write_rows
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.polar_library import import_polars
from structure_pretrain.library_builder.production import FINAL_SCHEMA
from structure_pretrain.library_builder.schemas import LibraryValidationError, load_build_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/library_builder"


def test_dedup_uses_full_canonical_string_and_read_only_baseline(tmp_path):
    db = tmp_path / "base.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE seen (smiles TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO seen VALUES ('CCO')")
    frozen = file_hash(db)
    rows = [
        {"canonical_smiles": text, "structure_id": str(i), "polar_id": "p", "combination_id": "c"}
        for i, text in enumerate(["CCO", "CCCC", "CCCC", "CCCN"])
    ]
    duplicates = []
    result = list(accept_unique(rows, db, duplicates))
    assert [r["canonical_smiles"] for r in result] == ["CCCC", "CCCN"]
    assert len(duplicates) == 2 and file_hash(db) == frozen


def test_preservation_and_no_refill_after_validation_failure(tmp_path, monkeypatch):
    baseline, cache, output = [tmp_path / name for name in ("v02", "cache", "v03")]
    baseline.mkdir()
    cache.mkdir()
    config = load_build_config(EXAMPLES / "demo_build.yaml")
    chains_path = EXAMPLES / "chains.tsv"
    chains = import_chains(chains_path, [])
    polars = sorted(import_polars(EXAMPLES / "polar_moieties.tsv", []), key=lambda p: p.polar_id)
    bank = build_combination_bank(chains, config)
    # Force the existing library to contain every permitted non-P background for
    # one head. Other heads still have unused combinations to propose.
    rows, polar_rows = [], []
    for i, polar in enumerate(polars):
        details = metadata(
            {**polar.row(), "polar_plausibility_tier": "CORE", "chemistry_family": "phosphate"}
        )
        polar_rows.append({**polar.row(), **details})
        for combo in [c for c in bank if c.linkage_family == "AA"][:3]:
            row = assemble(chains, combo, polar)
            row.update(
                details,
                polar_fragment_smiles=polar.polar_fragment_smiles,
                morgan_fp_2048=bytes(256),
                is_mandatory_pair=True,
            )
            row["structure_id"] = (
                "pl_" + hashlib.sha256(row["canonical_smiles"].encode()).hexdigest()
            )
            rows.append(row)
    write_rows(baseline / "structures.parquet", rows, FINAL_SCHEMA)
    pq.write_table(pa.Table.from_pylist(polar_rows), baseline / "polar_library.parquet")
    pq.write_table(
        pa.Table.from_pylist([c.row() for c in bank]), baseline / "chain_combinations.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist([c.row() for c in chains.values()]), baseline / "chain_library.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"polar_id": p.polar_id, "final_full_structure_count": 3} for p in polars]
        ),
        baseline / "polar_coverage.parquet",
    )
    write_rows(baseline / "special_cases.parquet", [], FINAL_SCHEMA)

    def dump(path, obj):
        path.write_text(json.dumps(obj), encoding="utf-8")

    dump(
        baseline / "input_hashes.json",
        {"chains": {"path": str(chains_path), "sha256": file_hash(chains_path)}},
    )
    (baseline / "build_config_resolved.yaml").write_text("frozen: true", encoding="utf-8")
    report = {
        "status": "complete",
        "final_unique_full_structures": len(rows),
        "structures_per_polar_max": 3,
        "canonical_duplicates": 0,
        "total_rejected": 0,
        "full_by_chemistry_family": {"phosphate": len(rows)},
        "by_structure_origin": dict(Counter(r["structure_origin"] for r in rows)),
        "by_plausibility_tier": {"CORE": len(rows)},
        "AA": len(rows),
        "OA": 0,
        "PE_PA": 0,
        "lyso": 0,
        "special_head_background_counts": {},
    }
    dump(baseline / "library_report.json", report)
    dump(
        baseline / "chain_coverage_report.json",
        chain_coverage(chains, Counter(t for r in rows for t in set(r["chain_tokens"]))),
    )
    digest = file_hash(baseline / "structures.parquet")
    dump(baseline / "verification_audit.json", {"status": "passed", "structures_sha256": digest})
    dump(cache / "fingerprint_manifest.json", {"structures_sha256": digest})
    np.savez(cache / "target_readiness_metrics.npz", max_similarity=np.ones(len(rows)))
    db = tmp_path / "baseline.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE seen (smiles TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO seen VALUES (?)", [(r["canonical_smiles"],) for r in rows])
    frozen = freeze_files([baseline, cache])
    calls = []

    def one_failure(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise LibraryValidationError("formula_invalid", "Simulated validation rejection")
        return assemble(*args, **kwargs)

    monkeypatch.setattr("structure_pretrain.library_builder.incremental.assemble", one_failure)
    result = build_increment(
        baseline,
        cache,
        output,
        db,
        chains_path,
        ROOT / "configs/library_build_expanded_v02.yaml",
        planned=3,
    )
    verify_frozen(frozen)
    assert result["increment_unique"] == 2
    assert result["increment_rejected"] == result["formula_rejected"] == 1
    assert result["final_unique_full_structures"] == len(rows) + result["increment_unique"]
    assert result["increment_planned"] == 3
    assert result["no_automatic_refill"] and result["PE_PA"] == 0
    assert verify_prefix(baseline / "structures.parquet", output / "structures.parquet") == len(
        rows
    )
    original_pairs = {(r["polar_id"], r["combination_id"]) for r in rows}
    added = pq.read_table(output / "increment_structures.parquet").to_pylist()
    assert all((r["polar_id"], r["combination_id"]) not in original_pairs for r in added)
    assert len({r["canonical_smiles"] for r in rows + added}) == len(rows) + len(added)
    assert (baseline / "input_hashes.json").read_bytes() == (
        output / "input_hashes.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="overwriting"):
        build_increment(baseline, cache, output, db, chains_path, "unused")


def test_append_all_columns_and_midrank_ties(tmp_path):
    old, extra, combined = [tmp_path / f"{name}.parquet" for name in ("old", "extra", "all")]
    table = pa.table({"a": ["one", "two"], "b": [[1, 2], []]})
    pq.write_table(table, old)
    pq.write_table(pa.table({"a": ["three"], "b": [[3]]}), extra)
    append_baseline(old, extra, combined)
    assert verify_prefix(old, combined) == 2
    assert pq.read_table(combined).to_pydict() == {
        "a": ["one", "two", "three"],
        "b": [[1, 2], [], [3]],
    }
    assert list(low_percentile([7, 7, 8, 15])) == [0.75, 0.75, 0.375, 0.125]
