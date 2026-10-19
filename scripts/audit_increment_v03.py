"""Independent full increment chemistry audit and exact preservation/overlap checks."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import make_combination, permitted
from structure_pretrain.library_builder.chain_library import import_chains
from structure_pretrain.library_builder.full_library_v02 import load_config_v02
from structure_pretrain.library_builder.incremental import read_json, verify_frozen, verify_prefix
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.production import validate_formula
from structure_pretrain.library_builder.schemas import ELEMENTS, PolarRule
from structure_pretrain.polar_plausibility.pipeline import dump_json


def audit(library, baseline):
    library, baseline = Path(library), Path(baseline)
    report = read_json(library / "library_report.json")
    inputs = read_json(library / "resolved_frozen_inputs.json")
    for entry in inputs.values():
        assert file_hash(Path(entry["path"])) == entry["sha256"]
    chains = import_chains(inputs["chains"]["path"], [])
    config, _ = load_config_v02(inputs["final_config"]["path"])
    selected = pq.read_table(library / "increment_selection.parquet").to_pylist()
    selections = {(r["polar_id"], r["combination_id"]) for r in selected}
    desired = {r["polar_id"] for r in selected}
    polars, details = {}, {}
    for batch in pq.ParquetFile(library / "polar_library.parquet").iter_batches(batch_size=500):
        for row in batch.to_pylist():
            if row["polar_id"] in desired:
                polars[row["polar_id"]] = PolarRule(
                    **{k: row[k] for k in PolarRule.__dataclass_fields__}
                )
                details[row["polar_id"]] = row
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    canonical, pairs = set(), set()
    families, origins, linkages = Counter(), Counter(), Counter()
    number = 0
    for batch in pq.ParquetFile(library / "increment_structures.parquet").iter_batches(
        batch_size=200
    ):
        for row in batch.to_pylist():
            pair = row["polar_id"], row["combination_id"]
            assert pair in selections and pair not in pairs
            assert row["canonical_smiles"] not in canonical
            pairs.add(pair)
            canonical.add(row["canonical_smiles"])
            combo = make_combination(tuple(row["chain_tokens"]), chains)
            assert permitted(combo, polars[row["polar_id"]], config)
            assert combo.linkage_family not in {"PA", "P"}
            fresh = assemble(chains, combo, polars[row["polar_id"]])
            validate_formula(row)
            for field in (
                "canonical_smiles",
                "neutral_formula",
                "inchikey",
                *[f"neutral_formula_{e}" for e in ELEMENTS],
            ):
                assert row[field] == fresh[field]
            assert (
                row["structure_id"]
                == "pl_" + hashlib.sha256(row["canonical_smiles"].encode()).hexdigest()
            )
            assert row["morgan_fp_2048"] == DataStructs.BitVectToBinaryText(
                fp.GetFingerprint(Chem.MolFromSmiles(row["canonical_smiles"]))
            )
            for field in (
                "source_id",
                "source_cids",
                "parent_source_cids",
                "structure_origin",
                "polar_fragment_smiles",
                "polar_plausibility_tier",
                "chemistry_family",
            ):
                assert row[field] == details[row["polar_id"]][field]
            families[row["chemistry_family"]] += 1
            origins[row["structure_origin"]] += 1
            linkages["lyso" if row["is_lyso"] else row["linkage_family"]] += 1
            number += 1
        if number % 1000 == 0:
            print(
                f"independent increment graph/formula/fingerprint audit "
                f"{number}/{report['increment_unique']}",
                flush=True,
            )
    scanned = 0
    for batch in pq.ParquetFile(baseline / "structures.parquet").iter_batches(
        batch_size=10000, columns=["canonical_smiles", "polar_id", "combination_id"]
    ):
        for row in batch.to_pylist():
            assert row["canonical_smiles"] not in canonical
            assert (row["polar_id"], row["combination_id"]) not in pairs
        scanned += batch.num_rows
    assert number == report["increment_unique"]
    assert dict(families) == report["increment_by_chemistry_family"]
    assert dict(origins) == report["increment_by_structure_origin"]
    assert dict(linkages) == report["increment_by_linkage"]
    assert verify_prefix(baseline / "structures.parquet", library / "structures.parquet") == scanned
    assert pq.ParquetFile(library / "structures.parquet").metadata.num_rows == scanned + number
    frozen = read_json(library / "baseline_preservation_manifest.json")
    verify_frozen(frozen)
    assert file_hash(library / "input_hashes.json") == file_hash(baseline / "input_hashes.json")
    result = {
        "status": "passed",
        "baseline_records_preserved_all_columns": scanned,
        "all_increment_graphs_independently_reassembled": number,
        "canonical_overlap_with_actual_baseline_parquet": 0,
        "reused_head_chain_pairs": 0,
        "increment_internal_canonical_duplicates": 0,
        "all_increment_sanitize_formula_elements_valence_and_fingerprints_checked": True,
        "all_frozen_files_and_original_inputs_rehashed": True,
        "all_original_input_hashes_preserved": True,
        "no_P_increment": True,
        "structures_sha256": file_hash(library / "structures.parquet"),
        "training_performed": False,
        "dataset_split_created": False,
    }
    dump_json(library / "verification_audit.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", default="data/structure_library_v0.3")
    p.add_argument("--baseline", default="data/structure_library_v0.2")
    a = p.parse_args()
    print(json.dumps(audit(a.library, a.baseline), indent=2), flush=True)
