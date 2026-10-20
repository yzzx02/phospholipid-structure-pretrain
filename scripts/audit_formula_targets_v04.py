"""CPU round trip of the frozen validation targets through the unchanged Builder."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from structure_pretrain.data.prepare_v03 import atom_counts, read_json
from structure_pretrain.data.real_structure_dataset import ShardedRowStore
from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.chain_combinations import make_combination, permitted
from structure_pretrain.library_builder.schemas import BuildConfig
from structure_pretrain.library_builder.validation import attachment
from structure_pretrain.polar_only_v04.chemistry import derive_polar_formula
from structure_pretrain.polar_only_v04.formula_beam_eval import EXPECTED_COHORT, EXPERIMENT
from structure_pretrain.polar_only_v04.prepare import LIBRARY, SAMPLES, V03, row_hash
from structure_pretrain.training.quality_v03 import load_builder_rules
from structure_pretrain.utils.formula import ELEMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT)
    args = parser.parse_args()
    output = args.experiment_root / "target_roundtrip_audit.json"
    if output.exists():
        raise ValueError("Existing audit is protected")
    rows = np.load(SAMPLES / "diagnostic_512.npy")
    if len(rows) != 512 or row_hash(rows) != EXPECTED_COHORT:
        raise ValueError("Frozen validation cohort changed")
    groups = np.load(V03 / "polar_groups.npy", mmap_mode="r")
    split = np.load(V03 / "row_splits.npy", mmap_mode="r")
    chain_labels = np.load(V03 / "chain_labels.npy", mmap_mode="r")
    formulas = np.load(V03 / "formulas.npy", mmap_mode="r")
    root = Path("data/private/pretraining_v04")
    chain_counts = np.load(root / "chain_counts.npy")
    scaffold = np.load(root / "scaffold_counts.npy")
    chain_vocab = read_json(V03 / "chain_vocab.json")["tokens"]
    texts = pq.read_table(V03 / "polar_groups.parquet", columns=["canonical_polar_smiles"])[
        "canonical_polar_smiles"
    ].to_pylist()
    if len(np.unique(groups[rows])) != 512 or not (split[rows] == 1).all():
        raise ValueError("Distinct validation Polar groups required")
    chains, heads = load_builder_rules(V03, LIBRARY)
    store = ShardedRowStore(V03)
    errors, multi_P, connected_P_not_first = [], 0, 0
    for index, row in enumerate(map(int, rows)):
        try:
            polar = texts[int(groups[row])]
            mol = parse_molecule(polar)
            expected = derive_polar_formula(
                formulas[row], chain_labels[row], chain_counts, scaffold
            )
            if not np.array_equal(atom_counts(mol), expected):
                raise ValueError("Polar_formula_mismatch_including_H")
            _, neighbor = attachment(mol, 1)
            phosphorus = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 15]
            if neighbor not in phosphorus:
                raise ValueError("Mapped_neighbor_is_not_P")
            multi_P += int(len(phosphorus) > 1)
            connected_P_not_first += int(neighbor != phosphorus[0])
            combo = make_combination(
                tuple(chain_vocab[int(t)] for t in chain_labels[row] if t), chains
            )
            rule = heads[polar]
            if not permitted(combo, rule, BuildConfig()):
                raise ValueError("Target_chain_combination_rejected")
            full = assemble(chains, combo, rule)
            if not np.array_equal(
                [full[f"neutral_formula_{element}"] for element in ELEMENTS], formulas[row]
            ):
                raise ValueError("Complete_neutral_formula_mismatch_including_H")
            if full["canonical_smiles"] != store.row(row)["canonical_full_smiles"]:
                raise ValueError("Complete_graph_roundtrip_mismatch")
        except (ValueError, KeyError) as exc:
            errors.append({"cohort_index": index, "reason": str(exc)})
    result = {
        "passed": not errors,
        "device": "cpu",
        "targets": 512,
        "distinct_validation_Polar_groups": 512,
        "complete_formula_and_graph_roundtrip_passed": 512 - len(errors),
        "multi_P_targets": multi_P,
        "mapped_P_differs_from_first_P_targets": connected_P_not_first,
        "cohort_rows_sha256": EXPECTED_COHORT,
        "errors": errors,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if errors:
        raise ValueError("Trusted target roundtrip audit failed")


if __name__ == "__main__":
    main()
