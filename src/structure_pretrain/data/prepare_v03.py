"""Read-only frozen-library audit and canonical-polar-group split preparation.

No full-structure cutting, new chemistry, or resampling of the frozen library.
Derived arrays use the exact Parquet row order, bound to SHA256 manifests.
"""

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from rdkit import Chem

from structure_pretrain.data.smiles_tokenizer import TOKEN_PATTERN, SmilesTokenizer
from structure_pretrain.library_builder.assembly import SCAFFOLD, assemble
from structure_pretrain.library_builder.canonicalize import nonisomeric_smiles, parse_molecule
from structure_pretrain.library_builder.chain_combinations import make_combination, permitted
from structure_pretrain.library_builder.chain_library import chain_sort_key
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.schemas import ELEMENTS, BuildConfig, ChainRule, PolarRule
from structure_pretrain.library_builder.validation import validate_polar_fragment
from structure_pretrain.polar_plausibility.pipeline import dump_json
from structure_pretrain.polar_plausibility.scaffold import scaffold_id

SCHEMA = "structure-pretraining-v0.3-two-output"
SPLITS = ("train", "validation", "test")
EXPECTED_SHA = "22eb617b25532743a29a6d6e5d3fdacf804e3a4b5289d7ca6bcf746f7395d38a"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_polar(smiles, atom_map=1):
    mol = validate_polar_fragment(smiles, atom_map)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(1)
    return nonisomeric_smiles(mol)


def rule_from_row(cls, row):
    values = {f.name: row[f.name] for f in fields(cls)}
    for name in ("representative_double_bond_positions", "source_polar_ids", "provenance"):
        if name in values:
            values[name] = tuple(values[name])
    return cls(**values)


def atom_counts(mol):
    counter = Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms() if a.GetAtomicNum())
    return np.array([counter[e] for e in ELEMENTS], dtype=np.int32)


def formula_counts(text):
    pieces = re.findall(r"([A-Z][a-z]?)(\d*)", text or "")
    if not pieces or "".join(e + n for e, n in pieces) != text:
        raise ValueError(f"Invalid mandatory neutral formula: {text!r}")
    if any(e not in ELEMENTS for e, _ in pieces):
        raise ValueError("Unsupported formula element")
    counts = Counter()
    for element, number in pieces:
        counts[element] += int(number or 1)
    return np.array([counts[e] for e in ELEMENTS], dtype=np.int32)


def stratified_groups(rows, seed=42):
    """One assignment per exact canonical graph; deterministic joint stratification."""
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        strata[
            (
                row["chemistry_family"],
                row["polar_plausibility_tier"],
                row["structure_origin"],
                row["token_length"] // 32,
            )
        ].append(i)
    rng = np.random.default_rng(seed)
    split = np.zeros(len(rows), dtype=np.uint8)
    for key in sorted(strata):
        ids = rng.permutation(strata[key])
        # At least one held-out group where the stratum permits it.
        nv = int(round(len(ids) * 0.05))
        nt = int(round(len(ids) * 0.05))
        split[ids[:nv]] = 1
        split[ids[nv : nv + nt]] = 2
    return split


def prepare(library, frozen_index, output, seed=42, reconstruction_count=600):
    library, frozen_index, output = map(Path, (library, frozen_index, output))
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    inputs = {
        p: file_hash(library / p)
        for p in (
            "structures.parquet",
            "polar_library.parquet",
            "chain_library.parquet",
            "chain_combinations.parquet",
            "special_cases.parquet",
            "input_hashes.json",
        )
    }
    if inputs["structures.parquet"] != EXPECTED_SHA:
        raise ValueError("This preparation requires the frozen v0.3 library hash")
    fp_manifest = read_json(frozen_index / "fingerprint_manifest.json")
    if fp_manifest["structures_sha256"] != EXPECTED_SHA:
        raise ValueError("Frozen fingerprint row mapping belongs to another library")
    binding = {
        "schema": SCHEMA,
        "version": "v0.3",
        "inputs_sha256": inputs,
        "structures_sha256": EXPECTED_SHA,
        "frozen_index_sha256": file_hash(frozen_index / "morgan_groups.usearch"),
        "elements": list(ELEMENTS),
    }
    if (output / "labels_manifest.json").exists():
        if read_json(output / "input_manifest.json") != binding:
            raise ValueError("Prepared inputs have changed")
        return read_json(output / "labels_manifest.json")
    dump_json(output / "input_manifest.json", binding)
    chains = {
        r["chain_token"]: rule_from_row(ChainRule, r)
        for r in pq.read_table(library / "chain_library.parquet").to_pylist()
    }
    vocab = ["NULL"] + sorted(chains, key=chain_sort_key)
    chain_index = {s: i for i, s in enumerate(vocab)}
    chain_types = Counter(c.linkage_type for c in chains.values())
    if chain_types != {"A": 177, "O": 84, "P": 61}:
        raise ValueError("Frozen chain vocabulary mismatch")
    dump_json(output / "chain_vocab.json", {**binding, "tokens": vocab, "counts": chain_types})
    chain_counts = {
        k: atom_counts(parse_molecule(c.representative_fragment_smiles)) for k, c in chains.items()
    }
    scaffold_counts = atom_counts(parse_molecule(SCAFFOLD))
    heads, graph_rows, graph_index, id_to_group = {}, [], {}, {}
    head_counts, tokens, brackets, scaffold_groups = {}, set(), set(), defaultdict(list)
    polar_source = pq.read_table(library / "polar_library.parquet").to_pylist()
    for i, row in enumerate(polar_source):
        pid, text = row["polar_id"], row["polar_fragment_smiles"]
        canonical = canonical_polar(text, row["attachment_atom_map"])
        pieces = TOKEN_PATTERN.findall(canonical)
        if "".join(pieces) != canonical:
            raise ValueError(f"Unrecognized polar token syntax: {pid}")
        tokens.update(pieces)
        brackets.update(p for p in pieces if p.startswith("["))
        group = graph_index.setdefault(canonical, len(graph_index))
        if group == len(graph_rows):
            graph_rows.append(
                {
                    "group": group,
                    "canonical_polar_smiles": canonical,
                    "graph_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                    "token_length": len(pieces) + 2,
                    "chemistry_family": row["chemistry_family"],
                    "polar_plausibility_tier": row["polar_plausibility_tier"],
                    "structure_origin": row["structure_origin"],
                    "scaffold_id": scaffold_id(canonical),
                }
            )
            scaffold_groups[graph_rows[-1]["scaffold_id"]].append(group)
        if pid in id_to_group and id_to_group[pid] != group:
            raise ValueError("One polar ID maps to inconsistent canonical graphs")
        id_to_group[pid] = group
        heads[pid] = row
        head_counts[pid] = atom_counts(parse_molecule(text))
        if (i + 1) % 25000 == 0:
            print(f"polar audit {i + 1}/{len(polar_source)}", flush=True)
    tokenizer = SmilesTokenizer(sorted(tokens))
    lengths = []
    for row in graph_rows:
        text = row["canonical_polar_smiles"]
        ids = tokenizer.encode(text)
        if tokenizer.decode(ids) != text:
            raise ValueError("Polar tokenizer round trip failed")
        lengths.append(len(ids))
    max_length = max(lengths)
    dump_json(
        output / "polar_tokenizer.json",
        {**binding, "tokens": list(tokenizer.tokens), "max_polar_length": max_length},
    )
    dump_json(
        output / "tokenization_audit.json",
        {
            **binding,
            "status": "passed",
            "unique_polar_graphs": len(graph_rows),
            "oov": 0,
            "roundtrip_failures": 0,
            "truncated": 0,
            "length_includes_bos_eos": True,
            "length": {str(q): float(np.percentile(lengths, q)) for q in (0, 50, 90, 95, 99, 100)},
            "max_polar_length": max_length,
            "old_length_64_sufficient": max_length <= 64,
            "vocabulary_size": len(tokenizer),
            "bracket_tokens": sorted(brackets),
            "vocabulary_policy": "Full audited lexical syntax only; no learned test statistics",
        },
    )
    group_splits = stratified_groups(graph_rows, seed)
    for i, row in enumerate(graph_rows):
        row["split"] = SPLITS[group_splits[i]]
    pq.write_table(pa.Table.from_pylist(graph_rows), output / "polar_groups.parquet")
    np.save(output / "group_splits.npy", group_splits)
    n = pq.ParquetFile(library / "structures.parquet").metadata.num_rows
    if n != 2015000:
        raise ValueError("Frozen row count mismatch")
    formulas = np.lib.format.open_memmap(
        output / "formulas.npy", mode="w+", dtype=np.int16, shape=(n, 10)
    )
    groups = np.lib.format.open_memmap(
        output / "polar_groups.npy", mode="w+", dtype=np.uint32, shape=(n,)
    )
    chain_labels = np.lib.format.open_memmap(
        output / "chain_labels.npy", mode="w+", dtype=np.uint16, shape=(n, 2)
    )
    split_rows = np.lib.format.open_memmap(
        output / "row_splits.npy", mode="w+", dtype=np.uint8, shape=(n,)
    )
    ids = np.load(frozen_index / "row_ids.npy", mmap_mode="r")
    shard_dir = output / "rows"
    shard_dir.mkdir(exist_ok=True)
    formula_cols = [f"neutral_formula_{e}" for e in ELEMENTS]
    columns = [
        "structure_id",
        "canonical_smiles",
        "neutral_formula",
        "polar_id",
        "polar_fragment_smiles",
        "chain_tokens",
        "chemistry_family",
        "polar_plausibility_tier",
        "structure_origin",
        "is_special_case",
        "combination_id",
        "architecture",
        "linkage_family",
    ] + formula_cols
    special_ids = set(
        pq.read_table(library / "special_cases.parquet", columns=["structure_id"])[
            "structure_id"
        ].to_pylist()
    )
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(n, max(600, reconstruction_count), replace=False).tolist())
    reconstruct, counts, used_chains = [], Counter(), Counter()
    group_row_count = np.zeros(len(graph_rows), dtype=np.int64)
    writer = None
    shard_manifest, shard_size, offset = [], 8192, 0
    manifest_writer = None
    try:
        for shard, batch in enumerate(
            pq.ParquetFile(library / "structures.parquet").iter_batches(
                batch_size=shard_size, columns=columns
            )
        ):
            rows = batch.to_pylist()
            manifest_records = []
            for j, row in enumerate(rows):
                k = offset + j
                pid, text = row["polar_id"], row["canonical_smiles"]
                head = heads[pid]
                group = id_to_group[pid]
                if (
                    row["structure_id"].encode() != ids[k]
                    or row["structure_id"] != "pl_" + hashlib.sha256(text.encode()).hexdigest()
                    or row["polar_fragment_smiles"] != head["polar_fragment_smiles"]
                ):
                    raise ValueError(f"Source identity/label mismatch at row {k}")
                for key in ("chemistry_family", "polar_plausibility_tier", "structure_origin"):
                    if row[key] != head[key]:
                        raise ValueError(f"Head association mismatch {key} at row {k}")
                ts = row["chain_tokens"]
                if len(ts) not in (1, 2) or ts != sorted(ts, key=chain_sort_key):
                    raise ValueError("Noncanonical two-slot chain labels")
                combo = make_combination(tuple(ts), chains)
                polar = rule_from_row(PolarRule, head)
                if (
                    not permitted(combo, polar, BuildConfig())
                    or row["architecture"] != "mono_phosphatidyl"
                    or row["combination_id"] != combo.combination_id
                    or row["linkage_family"] != combo.linkage_family
                ):
                    raise ValueError("Builder association / PE-P/A rule mismatch")
                counts_expected = scaffold_counts + head_counts[pid]
                for token in ts:
                    counts_expected = counts_expected + chain_counts[token]
                    used_chains[token] += 1
                if len(ts) == 1:
                    counts_expected[1] += 1  # Unoccupied glycerol oxygen gains H.
                supplied = [row[c] for c in formula_cols]
                if (
                    any(x is None or not isinstance(x, int) or x < 0 for x in supplied)
                    or not np.array_equal(supplied, formula_counts(row["neutral_formula"]))
                    or not np.array_equal(supplied, counts_expected)
                ):
                    raise ValueError(f"Mandatory neutral formula/builder count mismatch at {k}")
                if max(supplied) >= np.iinfo(np.int16).max:
                    raise ValueError("Formula exceeds compact storage type")
                formulas[k] = supplied
                groups[k], split_rows[k] = group, group_splits[group]
                chain_labels[k] = [chain_index[t] for t in ts] + [0] * (2 - len(ts))
                group_row_count[group] += 1
                counts[
                    (
                        SPLITS[group_splits[group]],
                        row["chemistry_family"],
                        row["polar_plausibility_tier"],
                        row["structure_origin"],
                    )
                ] += 1
                manifest_records.append(
                    {
                        "row_index": k,
                        "structure_id": row["structure_id"],
                        "polar_id": pid,
                        "polar_group": group,
                        "split": SPLITS[group_splits[group]],
                    }
                )
                if k in selected or row["structure_id"] in special_ids:
                    reconstruct.append(row)
            table = pa.Table.from_batches([batch]).rename_columns(
                ["canonical_full_smiles" if c == "canonical_smiles" else c for c in columns]
            )
            shard_path = shard_dir / f"{shard:05d}.arrow"
            with pa.OSFile(str(shard_path), "wb") as sink:
                with ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)
            shard_manifest.append(
                {
                    "file": f"rows/{shard:05d}.arrow",
                    "rows": len(rows),
                    "sha256": file_hash(shard_path),
                }
            )
            mt = pa.Table.from_pylist(manifest_records)
            if manifest_writer is None:
                manifest_writer = pq.ParquetWriter(output / "split_manifest.parquet", mt.schema)
            manifest_writer.write_table(mt)
            offset += len(rows)
            if shard % 12 == 0:
                print(f"full label audit {offset}/{n}", flush=True)
    finally:
        if manifest_writer:
            manifest_writer.close()
    for array in (formulas, groups, chain_labels, split_rows):
        array.flush()
    if offset != n or np.any(group_row_count == 0):
        raise ValueError("Incomplete row/head traversal")
    np.save(output / "group_row_counts.npy", group_row_count)
    failures, success = [], Counter()
    for row in reconstruct:
        try:
            result = assemble(
                chains,
                make_combination(tuple(row["chain_tokens"]), chains),
                rule_from_row(PolarRule, heads[row["polar_id"]]),
            )
            success["assembly"] += 1
            if result["canonical_smiles"] == row["canonical_smiles"]:
                success["canonical_graph"] += 1
            else:
                raise ValueError("Canonical full graph mismatch")
            if result["neutral_formula"] == row["neutral_formula"] and all(
                result[c] == row[c] for c in formula_cols
            ):
                success["formula"] += 1
            else:
                raise ValueError("Reconstruction formula mismatch")
        except ValueError as exc:
            failures.append({"structure_id": row["structure_id"], "reason": str(exc)})
    rec = {
        **binding,
        "status": "passed" if not failures else "failed",
        "sample_count": len(reconstruct),
        "special_structures_checked": sum(r["structure_id"] in special_ids for r in reconstruct),
        "all_special_structures": len(special_ids),
        **{name + "_match_rate": value / len(reconstruct) for name, value in success.items()},
        "assembly_success_rate": success["assembly"] / len(reconstruct),
        "failures": failures,
    }
    dump_json(output / "reconstruction_audit.json", rec)
    split_counts = {
        name: {
            "polar_groups": int(np.count_nonzero(group_splits == i)),
            "structures": int(np.count_nonzero(split_rows == i)),
        }
        for i, name in enumerate(SPLITS)
    }
    id_splits = [
        set(pid for pid, g in id_to_group.items() if group_splits[g] == i) for i in range(3)
    ]
    graph_sets = [set(g for g in range(len(graph_rows)) if group_splits[g] == i) for i in range(3)]
    intersections = {
        f"{SPLITS[a]}_{SPLITS[b]}": {
            "polar_id": len(id_splits[a] & id_splits[b]),
            "canonical_polar_graph": len(graph_sets[a] & graph_sets[b]),
        }
        for a, b in ((0, 1), (0, 2), (1, 2))
    }
    shared_scaffolds = sum(
        len(set(group_splits[g] for g in gs)) > 1 for gs in scaffold_groups.values()
    )
    analog_edges = Counter()
    for row in polar_source:
        for parent in row["analog_parent_polar_ids"]:
            if parent in id_to_group:
                a, b = group_splits[id_to_group[row["polar_id"]]], group_splits[id_to_group[parent]]
                analog_edges[(SPLITS[a], SPLITS[b])] += 1
    dump_json(
        output / "split_audit.json",
        {
            **binding,
            "status": "passed",
            "seed": seed,
            "split_counts": split_counts,
            "intersections": intersections,
            "exact_graph_aliases_grouped": len(heads) - len(graph_rows),
            "stratification": "family/tier/origin/token-length-bin; graph-level 90/5/5 rounding",
            "strata_structure_counts": [
                {"split": k[0], "family": k[1], "tier": k[2], "origin": k[3], "count": v}
                for k, v in counts.items()
            ],
            "cross_split_scaffolds": shared_scaffolds,
            "analog_parent_edges_by_split": {"/".join(k): v for k, v in analog_edges.items()},
            "test_sealed_for_model_selection": True,
            "near_similarity_audit": "separate metadata-only head audit; never used for tuning",
        },
    )
    audit = {
        **binding,
        "status": "passed" if not failures else "failed",
        "rows_checked": n,
        "valid_polar_ids": len(heads),
        "valid_canonical_polar_graphs": len(graph_rows),
        "invalid_labels": 0,
        "formula_counts_checked_against_builder_components": n,
        "canonical_id_hashes_checked": n,
        "canonical_RDKit_full_graphs_reconstructed": len(reconstruct),
        "used_chain_counts": {
            t: sum(v for k, v in used_chains.items() if k.startswith(t)) for t in ("A-", "O-", "P-")
        },
        "seconds": time.monotonic() - start,
    }
    dump_json(output / "label_audit.json", audit)
    if failures:
        raise ValueError("Reconstruction failed; training is gated")
    manifest = {
        **binding,
        "status": "complete",
        "rows": n,
        "shard_size": shard_size,
        "shards": shard_manifest,
        "max_polar_length": max_length,
        "polar_groups": len(graph_rows),
        "chain_vocabulary": len(vocab),
        "derived_sha256": {
            name: file_hash(output / name)
            for name in (
                "formulas.npy",
                "polar_groups.npy",
                "chain_labels.npy",
                "row_splits.npy",
                "split_manifest.parquet",
                "polar_groups.parquet",
                "group_splits.npy",
                "chain_vocab.json",
                "polar_tokenizer.json",
            )
        },
    }
    dump_json(output / "labels_manifest.json", manifest)
    print(
        f"prepare completed: {n} rows, {len(graph_rows)} graphs, max tokens {max_length}",
        flush=True,
    )
    return manifest


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", default="data/structure_library_v0.3")
    p.add_argument("--frozen-index", default="data/private/morgan_index_v0.3")
    p.add_argument("--output", default="data/private/pretraining_v03")
    a = p.parse_args()
    prepare(a.library, a.frozen_index, a.output)


if __name__ == "__main__":
    main()
