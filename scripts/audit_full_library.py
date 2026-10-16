"""Independently audit final counts/formulas/provenance and reconstruct sampled full graphs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
import pyarrow.parquet as pq
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import make_combination
from structure_pretrain.library_builder.chain_library import import_chains
from structure_pretrain.library_builder.coverage import chain_coverage
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.production import validate_formula
from structure_pretrain.library_builder.schemas import ELEMENTS, PolarRule
from structure_pretrain.polar_plausibility.pipeline import dump_json


def audit_library(library, chains_path, sample_count=300):
    library = Path(library)
    source = pq.ParquetFile(library / "structures.parquet")
    report = json.loads((library / "library_report.json").read_text(encoding="utf-8"))
    hashes = json.loads((library / "input_hashes.json").read_text(encoding="utf-8"))
    for entry in hashes.values():
        assert file_hash(Path(entry["path"])) == entry["sha256"]
    chains = import_chains(chains_path, [])
    polar_rows = pq.read_table(library / "polar_library.parquet").to_pylist()
    polars = {
        r["polar_id"]: PolarRule(**{k: r[k] for k in PolarRule.__dataclass_fields__})
        for r in polar_rows
    }
    per_head = Counter({k: 0 for k in polars})
    tokens, families, linkage, origins, tiers, special = (Counter() for _ in range(6))
    rng = np.random.default_rng(42)
    sample = set(
        rng.choice(
            source.metadata.num_rows, min(sample_count, source.metadata.num_rows), replace=False
        ).tolist()
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    columns = [
        "structure_id",
        "canonical_smiles",
        "neutral_formula",
        "polar_id",
        "polar_fragment_smiles",
        "chain_tokens",
        "polar_family",
        "chemistry_family",
        "polar_plausibility_tier",
        "structure_origin",
        "source_cids",
        "is_special_case",
        "is_lyso",
        "linkage_family",
        "combination_id",
        "morgan_fp_2048",
        *[f"neutral_formula_{e}" for e in ELEMENTS],
    ]
    number = rebuilt = 0
    for batch in source.iter_batches(batch_size=2000, columns=columns):
        for row in batch.to_pylist():
            validate_formula(row)
            assert (
                row["structure_id"]
                == "pl_" + hashlib.sha256(row["canonical_smiles"].encode()).hexdigest()
            )
            assert row["polar_fragment_smiles"] == polars[row["polar_id"]].polar_fragment_smiles
            assert row["polar_plausibility_tier"] in {"CORE", "EXTENDED"}
            assert len(row["morgan_fp_2048"]) == 256
            if row["structure_origin"] == "generated_phosphate_analog":
                assert not row["source_cids"]
            if any(t.startswith("P-") for t in row["chain_tokens"]):
                assert row["polar_family"] == "PE" and row["linkage_family"] == "PA"
            per_head[row["polar_id"]] += 1
            tokens.update(set(row["chain_tokens"]))
            families[row["chemistry_family"]] += 1
            origins[row["structure_origin"]] += 1
            tiers[row["polar_plausibility_tier"]] += 1
            linkage["lyso" if row["is_lyso"] else row["linkage_family"]] += 1
            if row["is_special_case"]:
                special[row["polar_id"]] += 1
            if number in sample or row["is_special_case"] or row["linkage_family"] == "PA":
                combo = make_combination(tuple(row["chain_tokens"]), chains)
                fresh = assemble(chains, combo, polars[row["polar_id"]])
                assert fresh["canonical_smiles"] == row["canonical_smiles"]
                assert fresh["neutral_formula"] == row["neutral_formula"]
                from rdkit import Chem

                assert (
                    DataStructs.BitVectToBinaryText(
                        generator.GetFingerprint(Chem.MolFromSmiles(fresh["canonical_smiles"]))
                    )
                    == row["morgan_fp_2048"]
                )
                rebuilt += 1
            number += 1
        if number % 100000 == 0:
            print(f"independent full-library audit {number}/{source.metadata.num_rows}", flush=True)
    assert number == report["final_unique_full_structures"] == source.metadata.num_rows
    assert dict(families) == report["full_by_chemistry_family"]
    assert min(per_head.values()) == report["structures_per_polar_min"] >= 3
    assert median(per_head.values()) == report["structures_per_polar_median"]
    assert max(per_head.values()) == report["structures_per_polar_max"]
    assert dict(special) == report["special_head_background_counts"]
    assert (
        dict(origins) == report["by_structure_origin"]
        and dict(tiers) == report["by_plausibility_tier"]
    )
    for key, name in (("AA", "AA"), ("OA", "OA"), ("PA", "PE_PA"), ("lyso", "lyso")):
        assert linkage[key] == report[name]
    coverage = chain_coverage(chains, tokens)
    assert coverage["status"] == "passed"
    saved_coverage = json.loads(
        (library / "chain_coverage_report.json").read_text(encoding="utf-8")
    )
    assert coverage == saved_coverage
    audit = {
        "status": "passed",
        "all_full_records_checked": number,
        "independently_reassembled_full_graphs": rebuilt,
        "independent_random_sample_target": sample_count,
        "all_special_and_PA_graphs_reassembled": True,
        "all_formula_element_vectors_checked": True,
        "all_structure_id_hashes_checked": True,
        "generated_analog_source_CIDs_not_misattributed": True,
        "P_only_for_exact_PE": True,
        "P_coverage_hard_gate": False,
        "covered_polar_count": sum(n > 0 for n in per_head.values()),
        "chain_coverage": coverage["families"],
        "structures_sha256": file_hash(library / "structures.parquet"),
        "training_performed": False,
        "dataset_split_created": False,
    }
    dump_json(library / "verification_audit.json", audit)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", default="data/structure_library_v0.2")
    p.add_argument(
        "--chains", default="data/private/msdial_chain_review/msdial_derived_chains_v0.1.tsv"
    )
    a = p.parse_args()
    print(json.dumps(audit_library(a.library, a.chains), indent=2), flush=True)


if __name__ == "__main__":
    main()
