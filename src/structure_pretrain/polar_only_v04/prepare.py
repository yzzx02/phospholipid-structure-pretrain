"""Independent v0.4 chemistry audits and bounded, sharded fragment feature caches."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from rdkit import Chem

from structure_pretrain.data.prepare_v03 import atom_counts, read_json
from structure_pretrain.data.real_structure_dataset import ShardedRowStore
from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.chain_combinations import make_combination, permitted
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.schemas import BuildConfig
from structure_pretrain.polar_only_v04 import SCHEMA
from structure_pretrain.polar_only_v04.chemistry import (
    attach_known_chains,
    chainless_molecule,
    derive_polar_formula,
    fragment_count_inputs,
)
from structure_pretrain.polar_only_v04.features import (
    fragment_features,
    fragment_molecule,
    packed_fingerprint,
)
from structure_pretrain.training.quality_v03 import load_builder_rules

DEFAULT_ROOT = Path("data/private/pretraining_v04")
V03 = Path("data/private/pretraining_v03")
LIBRARY = Path("data/structure_library_v0.3")
SAMPLES = Path("outputs/pretraining_v04/samples")


def emit(value):
    print(json.dumps(value), flush=True)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def freeze_samples(groups, split, chains, head):
    SAMPLES.mkdir(parents=True, exist_ok=True)
    source = Path("outputs/midterm_v03")
    selected = {
        n: np.load(source / f"{n}.npy") for n in ("train_pool", "validation_pool", "diagnostic_512")
    }
    chain_code = chains[:, 0].astype(np.int64) * 323 + chains[:, 1]
    codes, counts = np.unique(chain_code[split == 0], return_counts=True)
    dominant = int(codes[counts.argmax()])
    eligible = np.flatnonzero((split == 0) & (chain_code == dominant))
    rng = np.random.default_rng(42)
    chosen, covered = [], set()
    for length in (min(head["token_length"]), max(head["token_length"])):
        matches = [int(r) for r in eligible if head["token_length"][int(groups[r])] == length]
        if matches:
            chosen.append(matches[0])
            covered.add(int(groups[matches[0]]))
    buckets = defaultdict(list)
    for row in rng.permutation(eligible):
        g = int(groups[row])
        buckets[f"{head['chemistry_family'][g]}/{head['token_length'][g] // 16}"].append(int(row))
    while len(chosen) < 32:
        for key in sorted(buckets):
            while buckets[key] and int(groups[buckets[key][-1]]) in covered:
                buckets[key].pop()
            if buckets[key]:
                row = buckets[key].pop()
                chosen.append(row)
                covered.add(int(groups[row]))
                if len(chosen) == 32:
                    break
    selected["overfit_32"] = np.asarray(chosen, np.int64)
    ids = np.load("data/private/morgan_index_v0.3/row_ids.npy", mmap_mode="r")
    metadata = {}
    for name, rows in selected.items():
        expected = 0 if name in ("train_pool", "overfit_32") else 1
        assert np.all(split[rows] == expected)
        path = SAMPLES / f"{name}.npy"
        if path.exists():
            np.testing.assert_array_equal(np.load(path), rows)
        else:
            np.save(path, rows)
        metadata[name] = {
            "count": len(rows),
            "rows_sha256": row_hash(rows),
            "target_ids_sha256": hashlib.sha256(
                b"\n".join(bytes(ids[r]) for r in rows)
            ).hexdigest(),
            "distinct_polar_graphs": len(np.unique(groups[rows])),
        }
    assert metadata["diagnostic_512"]["distinct_polar_graphs"] == 512
    assert metadata["overfit_32"]["distinct_polar_graphs"] == 32
    dump(
        SAMPLES / "manifest.json",
        {
            "schema": SCHEMA,
            "seed": 42,
            "cohorts": metadata,
            "overfit_chain_combination_code": dominant,
            "reuse_v03_target_ids_only_not_weights_or_fingerprints": True,
            "test_targets_used_for_training_or_selection": 0,
        },
    )
    return selected


def prepare_formula(root):
    root.mkdir(parents=True, exist_ok=True)
    binding = {
        "schema": SCHEMA,
        "source_labels_sha256": file_hash(V03 / "labels_manifest.json"),
        "source_split_sha256": file_hash(V03 / "row_splits.npy"),
        "source_polar_groups_sha256": file_hash(V03 / "polar_groups.npy"),
        "source_chain_library_sha256": file_hash(LIBRARY / "chain_library.parquet"),
        "source_structures_sha256": read_json(V03 / "labels_manifest.json")["structures_sha256"],
        "feature_definition": "glycerol+P-polar; occupied chain sites maps101/102; real lyso OH",
        "shard_size": 2048,
    }
    if (root / "source_binding.json").exists():
        assert read_json(root / "source_binding.json") == binding
    else:
        dump(root / "source_binding.json", binding)
    if (root / "formula_audit.json").exists():
        assert read_json(root / "formula_audit.json")["status"] == "passed"
        return
    groups = np.load(V03 / "polar_groups.npy", mmap_mode="r")
    split = np.load(V03 / "row_splits.npy", mmap_mode="r")
    labels = np.load(V03 / "chain_labels.npy", mmap_mode="r")
    head = pq.read_table(V03 / "polar_groups.parquet").to_pydict()
    samples = freeze_samples(groups, split, labels, head)
    chains, rules = load_builder_rules(V03, LIBRARY, emit)
    vocab = read_json(V03 / "chain_vocab.json")["tokens"]
    chain_counts, scaffold_counts = fragment_count_inputs(chains, vocab)
    np.save(root / "chain_counts.npy", chain_counts)
    np.save(root / "scaffold_counts.npy", scaffold_counts)
    head_counts = np.empty((len(head["group"]), 10), dtype=np.int16)
    for g, text in enumerate(head["canonical_polar_smiles"]):
        head_counts[g] = atom_counts(parse_molecule(text))
        if (g + 1) % 4096 == 0:
            emit({"stage": "head_fragment_counts", "heads": g + 1})
    np.save(root / "head_polar_formulas.npy", head_counts)
    formulas = np.load(V03 / "formulas.npy", mmap_mode="r")
    for start in range(0, len(groups), 65536):
        end = min(start + 65536, len(groups))
        derived = derive_polar_formula(
            formulas[start:end], labels[start:end], chain_counts, scaffold_counts
        )
        np.testing.assert_array_equal(derived, head_counts[groups[start:end]])
        emit({"stage": "all_row_formula_identity", "rows": end})
    store = ShardedRowStore(V03)
    ids = np.load("data/private/morgan_index_v0.3/row_ids.npy", mmap_mode="r")
    special = pq.read_table(LIBRARY / "special_cases.parquet", columns=["structure_id"])[
        "structure_id"
    ].to_pylist()
    special_rows = np.flatnonzero(np.isin(ids, np.asarray(special, dtype="S67")))
    initial = np.unique(
        np.r_[np.random.default_rng(42).choice(len(groups), 600, replace=False), special_rows]
    )
    assert len(initial) == 671 and len(special_rows) == 71
    # New, deterministic TRAIN-only independent reconstruction sample (not a test selection).
    eligible = np.flatnonzero(split == 0)
    buckets = defaultdict(list)
    for row in np.random.default_rng(20261019).permutation(eligible):
        g = int(groups[row])
        toks = [vocab[int(t)].split("-")[0] for t in labels[row] if t]
        buckets[
            f"{head['chemistry_family'][g]}/{'-'.join(toks)}/{head['token_length'][g] // 16}"
        ].append(int(row))
    independent = []
    while len(independent) < 10000:
        for key in sorted(buckets):
            if buckets[key]:
                independent.append(buckets[key].pop())
            if len(independent) == 10000:
                break

    def audit(rows, name):
        strata = Counter()
        for count, row_index in enumerate(rows, 1):
            row = store.row(int(row_index))
            g = int(groups[row_index])
            rule = rules[head["canonical_polar_smiles"][g]]
            combo = make_combination(tuple(row["chain_tokens"]), chains)
            assert permitted(combo, rule, BuildConfig())
            rebuilt = assemble(chains, combo, rule)
            assert rebuilt["canonical_smiles"] == row["canonical_full_smiles"]
            actual = atom_counts(parse_molecule(row["canonical_full_smiles"]))
            derived = derive_polar_formula(actual, labels[row_index], chain_counts, scaffold_counts)
            np.testing.assert_array_equal(derived, head_counts[g])
            fragment = chainless_molecule(rule.polar_fragment_smiles, len(combo.chain_tokens))
            text = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=False)
            fragment = fragment_molecule(text)
            expected = scaffold_counts + head_counts[g]
            expected = expected.copy()
            expected[1] += combo.is_lyso
            np.testing.assert_array_equal(atom_counts(fragment), expected)
            full_text, _, _, _ = attach_known_chains(fragment, combo.chain_tokens, chains)
            assert full_text == row["canonical_full_smiles"]
            strata[f"{head['chemistry_family'][g]}/{combo.linkage_family}"] += 1
            if count % 100 == 0:
                emit({"stage": name, "rows": count})
        return {
            "count": len(rows),
            "rows_sha256": row_hash(rows),
            "failures": 0,
            "family_linkage_counts": dict(strata),
        }

    first, extra = audit(initial, "roundtrip_671"), audit(independent, "roundtrip_10000")
    np.save(root / "roundtrip_671_rows_private.npy", initial)
    np.save(root / "roundtrip_10000_rows_private.npy", np.asarray(independent, np.int64))
    assert all(
        any(k.startswith(family + "/") for k in extra["family_linkage_counts"])
        for family in ("phosphate", "phosphonate", "polyphosphate")
    )
    assert np.all(split[samples["train_pool"]] == 0)
    dump(
        root / "formula_audit.json",
        {
            "schema": SCHEMA,
            "status": "passed",
            "all_frozen_rows_formula_verified": len(groups),
            "negative_or_mismatched_polar_formulas": 0,
            "initial_671_including_all_71_special": first,
            "independent_10000_train": extra,
            "dummy_H_convention": (
                "attached scaffold + attached chain fragments + attached polar; lyso +H"
            ),
            "same_polar_different_AA_OA_chains_share_normalized_graph": True,
            "lyso_has_real_vacant_OH_not_extra_dummy": True,
        },
    )


def prepare_features(root):
    if (root / "features_manifest.json").exists():
        return
    if read_json(root / "formula_audit.json")["status"] != "passed":
        raise ValueError("Formula gate")
    groups = np.load(V03 / "polar_groups.npy", mmap_mode="r")
    labels = np.load(V03 / "chain_labels.npy", mmap_mode="r")
    codes = groups.astype(np.int64) * 2 + (labels[:, 1] == 0)
    variants = np.unique(codes)
    np.save(root / "row_variants.npy", np.searchsorted(variants, codes).astype(np.int32))
    np.save(root / "variant_heads.npy", (variants // 2).astype(np.uint32))
    np.save(root / "variant_occupancy.npy", (2 - variants % 2).astype(np.uint8))
    head = pq.read_table(V03 / "polar_groups.parquet", columns=["canonical_polar_smiles"])[
        0
    ].to_pylist()
    counts = np.load(root / "head_polar_formulas.npy", mmap_mode="r")
    packed = np.lib.format.open_memmap(
        root / "chainless_fingerprints.npy", mode="w+", dtype=np.uint8, shape=(len(variants), 256)
    )
    folder = root / "fragments"
    folder.mkdir(exist_ok=True)
    schema = pa.schema(
        [
            ("variant", pa.uint32()),
            ("head", pa.uint32()),
            ("occupancy", pa.uint8()),
            ("fragment_smiles", pa.string()),
            ("atoms", pa.list_(pa.list_(pa.uint8()))),
            ("edges", pa.list_(pa.list_(pa.uint16()))),
            ("bonds", pa.list_(pa.list_(pa.uint8()))),
            ("site_roles", pa.list_(pa.uint8())),
        ]
    )
    shards = []
    for start in range(0, len(variants), 2048):
        records = []
        for v in range(start, min(start + 2048, len(variants))):
            g, occupancy = int(variants[v] // 2), int(2 - variants[v] % 2)
            text = Chem.MolToSmiles(
                chainless_molecule(head[g], occupancy), canonical=True, isomericSmiles=False
            )
            molecule = fragment_molecule(text)
            packed[v] = packed_fingerprint(molecule)
            records.append(
                {
                    "variant": v,
                    "head": g,
                    "occupancy": occupancy,
                    "fragment_smiles": text,
                    **fragment_features(molecule),
                }
            )
            assert np.all(counts[g] >= 0)
        path = folder / f"{start // 2048:05d}.arrow"
        with pa.OSFile(str(path), "wb") as sink:
            with ipc.new_file(sink, schema) as writer:
                writer.write_table(pa.Table.from_pylist(records, schema=schema))
        shards.append(
            {"file": str(path.relative_to(root)), "rows": len(records), "sha256": file_hash(path)}
        )
        packed.flush()
        emit(
            {
                "stage": "chainless_graph_and_Morgan_cache",
                "variants": start + len(records),
                "total": len(variants),
            }
        )
    dump(
        root / "features_manifest.json",
        {
            "schema": SCHEMA,
            "status": "complete",
            "variants": len(variants),
            "shard_size": 2048,
            "shards": shards,
            "fingerprints_sha256": file_hash(root / "chainless_fingerprints.npy"),
            "row_variants_sha256": file_hash(root / "row_variants.npy"),
            "radius": 2,
            "bits": 2048,
            "old_full_molecule_fingerprints_used": False,
            "formula_audit_sha256": file_hash(root / "formula_audit.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=["chemistry", "features", "all"], default="all")
    args = parser.parse_args()
    prepare_formula(args.root)
    if args.stage != "chemistry":
        prepare_features(args.root)
    if args.stage == "all":
        from structure_pretrain.polar_only_v04.retrieval import prepare_retrieval

        prepare_retrieval(args.root)


if __name__ == "__main__":
    main()
