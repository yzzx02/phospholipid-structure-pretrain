"""Unrepaired generation diagnostics and frozen-builder acceptance stages."""

from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.data.prepare_v03 import canonical_polar, rule_from_row
from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.chain_combinations import make_combination, permitted
from structure_pretrain.library_builder.schemas import ELEMENTS, BuildConfig, ChainRule, PolarRule
from structure_pretrain.library_builder.validation import attachment

CATEGORIES = (
    "no_eos",
    "smiles_syntax",
    "attachment",
    "polar_or_builder_chemistry",
    "chain_combination",
    "formula_mismatch",
    "formula_matches_wrong_structure",
    "correct_structure",
)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)


def polar_checks(text, eos):
    result = {
        "eos": bool(eos),
        "syntax_valid": False,
        "rdkit_valid": False,
        "attachment_valid": False,
        "polar_valid": False,
    }
    if not eos:
        return {**result, "category": "no_eos", "reason": "length_exhausted"}
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(text, sanitize=False)
    if molecule is None or not molecule.GetNumAtoms():
        return {**result, "category": "smiles_syntax", "reason": "invalid_smiles"}
    result["syntax_valid"] = True
    try:
        parse_molecule(text)
        result["rdkit_valid"] = True
    except ValueError:
        pass
    try:
        _, neighbor = attachment(molecule, 1)
        if molecule.GetAtomWithIdx(neighbor).GetAtomicNum() != 15:
            raise ValueError("polar_boundary_invalid")
        result["attachment_valid"] = True
    except ValueError as exc:
        return {
            **result,
            "category": "attachment",
            "reason": getattr(exc, "reason", "polar_boundary_invalid"),
        }
    try:
        result["canonical_polar"] = canonical_polar(text)
        result["polar_valid"] = True
        return result
    except ValueError as exc:
        return {
            **result,
            "category": "polar_or_builder_chemistry",
            "reason": getattr(exc, "reason", "invalid_polar_chemistry"),
        }


def score_candidate(
    text,
    eos,
    chain_tokens,
    chains,
    known_heads,
    target_formula,
    target_polar,
    target_full,
    template_polars,
):
    result = polar_checks(text, eos)
    result.update(
        assembled=False,
        formula_match=False,
        full_exact=False,
        polar_exact=False,
        template_copy=False,
        morgan_similarity=0.0,
    )
    if not result["polar_valid"]:
        return result
    graph = result["canonical_polar"]
    result["polar_exact"] = graph == target_polar
    result["template_copy"] = graph in template_polars
    rule = known_heads.get(graph)
    if rule is None:
        rule = PolarRule(
            "generated", "model", "free_generation", text, text, 1, "generated", True, True, False
        )
    try:
        combo = make_combination(tuple(t for t in chain_tokens if t != "NULL"), chains)
        if not permitted(combo, rule, BuildConfig()):
            raise ValueError("disallowed_chain_polar_combination")
    except (ValueError, KeyError):
        return {**result, "category": "chain_combination", "reason": "unsupported_or_disallowed"}
    try:
        full = assemble(chains, combo, rule)
    except (ValueError, KeyError) as exc:
        return {
            **result,
            "category": "polar_or_builder_chemistry",
            "reason": getattr(exc, "reason", "assembly_failed"),
        }
    result.update(assembled=True, full_smiles=full["canonical_smiles"])
    counts = [full[f"neutral_formula_{e}"] for e in ELEMENTS]
    result["formula_match"] = bool(np.array_equal(counts, target_formula))
    result["full_exact"] = full["canonical_smiles"] == target_full
    result["morgan_similarity"] = DataStructs.TanimotoSimilarity(
        MORGAN.GetFingerprint(parse_molecule(full["canonical_smiles"])),
        MORGAN.GetFingerprint(parse_molecule(target_full)),
    )
    if not result["formula_match"]:
        result["category"] = "formula_mismatch"
    elif not (result["polar_exact"] and result["full_exact"]):
        result["category"] = "formula_matches_wrong_structure"
    else:
        result["category"] = "correct_structure"
    return result


def aggregate(rows):
    n = len(rows)
    if not n:
        raise ValueError("Empty evaluation cohort")
    counts = Counter(r["raw"]["category"] for r in rows)
    metrics = {
        "evaluated_targets": n,
        "failure_categories": {
            c: {"count": counts[c], "fraction": counts[c] / n} for c in CATEGORIES
        },
        "failure_reasons": dict(
            Counter(r["raw"].get("reason", r["raw"]["category"]) for r in rows)
        ),
    }
    for output, key in (
        ("eos_rate", "eos"),
        ("syntax_parse_rate", "syntax_valid"),
        ("rdkit_parse_rate", "rdkit_valid"),
        ("topological_attachment_rate", "attachment_valid"),
        ("valid_polar_rate", "polar_valid"),
        ("exact_polar_graph_rate", "polar_exact"),
        ("template_copy_rate", "template_copy"),
        ("assembly_success_rate", "assembled"),
        ("full_formula_match_rate", "formula_match"),
        ("full_structure_exact_rate", "full_exact"),
    ):
        metrics[output] = sum(r["raw"][key] for r in rows) / n
    metrics["attachment_valid_rate"] = metrics["valid_polar_rate"]  # legacy audited-polar meaning
    metrics["top5_polar_recall"] = sum(r["top5_polar_recalled"] for r in rows) / n
    metrics["chain_count_accuracy"] = sum(r["chain_count_correct"] for r in rows) / n
    metrics["chain_set_exact_rate"] = sum(r["chain_set_exact"] for r in rows) / n
    metrics["non_null_chain_accuracy"] = sum(r["nonnull_correct"] for r in rows) / max(
        sum(r["nonnull_total"] for r in rows), 1
    )
    metrics["target_morgan_similarity"] = sum(r["raw"]["morgan_similarity"] for r in rows) / n
    assembled = [r for r in rows if r["raw"]["assembled"]]
    metrics["target_morgan_similarity_among_assembled"] = (
        sum(r["raw"]["morgan_similarity"] for r in assembled) / len(assembled)
        if assembled
        else None
    )
    lengths = np.array([r["generated_length"] for r in rows])
    metrics["length_exhausted_rate"] = 1 - metrics["eos_rate"]
    metrics["generated_length"] = {
        "definition": "tokens excluding BOS, including actual EOS when present",
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "p50_p90_p95_p99": np.quantile(lengths, [0.5, 0.9, 0.95, 0.99]).tolist(),
        "histogram": dict(sorted(Counter(lengths.tolist()).items())),
    }
    selected = [r["post_filter_selected"] for r in rows if r["post_filter_selected"] is not None]
    hits = sum(r["full_exact"] for r in selected)
    metrics["optional_formula_filter"] = {
        "scope": "same chain argmax, up to five raw polar beams; no repair",
        "selected_targets": len(selected),
        "coverage": len(selected) / n,
        "full_exact_count": hits,
        "overall_full_exact": hits / n,
        "full_exact_among_selected": hits / len(selected) if selected else None,
    }
    return metrics


def load_builder_rules(prepared, library, progress=lambda x: None):
    """Reuse frozen canonical labels instead of re-canonicalizing 195k head SMILES."""
    prepared, library = Path(prepared), Path(library)
    groups = np.load(prepared / "polar_groups.npy", mmap_mode="r")
    head_table = pq.read_table(
        prepared / "polar_groups.parquet", columns=["canonical_polar_smiles"]
    )
    canonicals = head_table["canonical_polar_smiles"].to_pylist()
    polar_to_graph, offset = {}, 0
    for part in pq.ParquetFile(library / "structures.parquet").iter_batches(
        batch_size=65536, columns=["polar_id"]
    ):
        _, positions = np.unique(groups[offset : offset + len(part)], return_index=True)
        ids = part.column(0)
        for pos in positions:
            polar_to_graph[ids[int(pos)].as_py()] = canonicals[int(groups[offset + pos])]
        offset += len(part)
        progress({"stage": "head_lookup", "structure_rows": offset})
    assert offset == len(groups)
    polars = {}
    columns = [f.name for f in fields(PolarRule)]
    for part in pq.ParquetFile(library / "polar_library.parquet").iter_batches(
        batch_size=4096, columns=columns
    ):
        for row in part.to_pylist():
            if row["polar_id"] in polar_to_graph:
                polars[polar_to_graph[row["polar_id"]]] = rule_from_row(PolarRule, row)
        progress({"stage": "head_rules", "heads": len(polars)})
    assert len(polars) == len(canonicals)
    chains = {
        r["chain_token"]: rule_from_row(ChainRule, r)
        for r in pq.read_table(library / "chain_library.parquet").to_pylist()
    }
    return chains, polars
