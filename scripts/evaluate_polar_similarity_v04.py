"""CPU-only, model-ranked similarity and paired template gains on diagnostic512."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from analyze_polar_fp50k import graph_info
from rdkit import DataStructs
from report_formula_beam_cuda_v04 import aggregate_jobs

from structure_pretrain.data.prepare_v03 import atom_counts, canonical_polar, read_json
from structure_pretrain.data.real_structure_dataset import ShardedRowStore
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.constrained_decode import rank_unique_accepted
from structure_pretrain.polar_only_v04.dataset import PolarDataset
from structure_pretrain.polar_only_v04.formula_beam_eval import EXPECTED_COHORT
from structure_pretrain.polar_only_v04.fp50k_data import FEATURES, get_contract
from structure_pretrain.polar_only_v04.fp50k_data import ROOT as TRAIN_ROOT
from structure_pretrain.polar_only_v04.prepare import LIBRARY, SAMPLES, V03, row_hash
from structure_pretrain.training.quality_v03 import load_builder_rules, score_candidate

ROOT = Path("outputs/polar_similarity_eval_v04_20261020")
N_VALUES = (1, 3, 5, 10)
EPS = 1e-12
MODELS = ("20k_3000", "50k_3000", "50k_5000")
CHECKPOINTS = {
    "20k_3000": Path("outputs/pretraining_v04/runs/fp_smoke_3000/checkpoint/step_003000.pt"),
    "50k_3000": TRAIN_ROOT / "train_B_3000/checkpoint/step_003000.pt",
    "50k_5000": TRAIN_ROOT / "train_C_5000/checkpoint/step_005000.pt",
}


def write(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def ratio(count, denominator):
    return {
        "count": int(count),
        "denominator": int(denominator),
        "fraction": float(count / denominator) if denominator else None,
    }


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "quantiles_0_10_25_50_75_90_100": np.quantile(
            values, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        ).tolist()
        if len(values)
        else [],
    }


def topn_similarities(model_ranked_similarities):
    """Prefix first, maximum second. Never reorder by the unknown target."""
    return {str(n): max(model_ranked_similarities[:n], default=0.0) for n in N_VALUES}


def gain_summary(generated, template):
    delta = np.asarray(generated) - np.asarray(template)
    result = {
        **distribution(delta),
        "positive": ratio((delta > EPS).sum(), len(delta)),
        "zero": ratio((np.abs(delta) <= EPS).sum(), len(delta)),
        "negative": ratio((delta < -EPS).sum(), len(delta)),
    }
    if len(delta):
        draws = np.random.default_rng(42).integers(0, len(delta), (2000, len(delta)))
        result["paired_diagnostic_bootstrap_mean_95_percentile"] = np.quantile(
            delta[draws].mean(1), [0.025, 0.975]
        ).tolist()
    return result


def sim(a, b):
    return float(DataStructs.TanimotoSimilarity(graph_info(a)[0], graph_info(b)[0]))


def thresholds(values):
    values = np.asarray(values)
    return {
        str(t): ratio((values >= t - EPS).sum(), len(values)) for t in (0.5, 0.6, 0.7, 0.8, 0.9)
    } | {"1.0": ratio((np.abs(values - 1) <= EPS).sum(), len(values))}


def sample_summary(rows):
    n = len(rows)
    covered = [r for r in rows if r["candidate_count"]]
    result = {
        "targets": n,
        "coverage": ratio(len(covered), n),
        "no_candidate": ratio(n - len(covered), n),
        "topn": {},
        "template_top1": distribution([r["template_top1"] for r in rows]),
        "template_best": distribution([r["template_best"] for r in rows]),
    }
    for label, subset in (("all_targets", rows), ("covered_targets", covered)):
        result[label] = {}
        for k in map(str, N_VALUES):
            values = [r["s"][k] for r in subset]
            result[label][k] = {
                "similarity": distribution(values),
                "thresholds": thresholds(values),
                "exact": ratio(sum(r["exact"][k] for r in subset), len(subset)),
                "tanimoto1_without_exact": ratio(
                    sum(abs(r["s"][k] - 1) <= EPS and not r["exact"][k] for r in subset),
                    len(subset),
                ),
            }
            if k in ("1", "5", "10"):
                result[label][k]["delta_vs_template_best"] = gain_summary(
                    values, [r["template_best"] for r in subset]
                )
        result[label]["all_strict_candidates"] = {
            "similarity": distribution([r["s_all"] for r in subset]),
            "thresholds": thresholds([r["s_all"] for r in subset]),
        }
    return result


def records_from(paths):
    records = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    records.sort(key=lambda r: r["cohort_index"])
    if [r["cohort_index"] for r in records] != list(range(512)):
        raise ValueError("Every case needs 512 complete, disjoint targets")
    return records


def cases(rows, allow_incomplete):
    result = {}
    old = Path("outputs/formula_constrained_v04_rtx4070_20261020")
    repaired = Path(str(old) + "_r1")
    previous, _ = aggregate_jobs(repaired, rows, old)
    for case in previous:
        if (case["encoder"], case["mode"]) != ("fingerprint", "elements_grammar") or case[
            "beam_size"
        ] not in (20, 50):
            continue
        assert case["status"] == "complete_512"
        paths = [
            Path("outputs") / j["origin_experiment"] / j["name"] / "predictions_private.jsonl"
            for j in case["jobs"]
        ]
        assert case["inputs"]["checkpoint_sha256"] == file_hash(CHECKPOINTS["20k_3000"])
        result[f"20k_3000_b{case['beam_size']}"] = (records_from(paths), paths)
    folders = {
        "50k_3000_b20": "eval_B_3000_beam20_512",
        "50k_3000_b50": "eval_similarity_3000_beam50_512",
        "50k_5000_b20": "eval_C_5000_beam20_512",
        "50k_5000_b50": "eval_C_5000_beam50_512",
    }
    _, _, contract = get_contract(TRAIN_ROOT, True)
    assert contract == read_json(TRAIN_ROOT / "cpu_gate.json")["contract"]
    for key, folder in folders.items():
        run = TRAIN_ROOT / folder
        if not (run / "host_monitor.json").exists() and allow_incomplete:
            continue
        host, metric = read_json(run / "host_monitor.json"), read_json(run / "metrics.json")
        assert host["stop_reason"] is None and host["child_exit_code"] == 0
        assert host["protected_checkpoints_unchanged"]
        model = key.rsplit("_b", 1)[0]
        assert metric["checkpoint_sha256"] == file_hash(CHECKPOINTS[model])
        assert metric["count"] == 512 and metric["cohort_rows_sha256"] == EXPECTED_COHORT
        paths = [run / "predictions_private.jsonl"]
        result[key] = (records_from(paths), paths)
    return result


def auxiliary(records, private):
    raw = [c for r in records for c in r["raw_candidates"]]
    strict = [c for r in records for c in r["accepted_unique"]]
    n = len(records)
    return {
        "raw_candidates": len(raw),
        "strict_unique_candidates": len(strict),
        "raw_RDKit_valid": ratio(sum(c["rdkit_valid"] for c in raw), len(raw)),
        "raw_Polar_chemistry_valid": ratio(sum(c["polar_valid"] for c in raw), len(raw)),
        "raw_full_neutral_formula_matches": ratio(sum(c["formula_match"] for c in raw), len(raw)),
        "raw_top1_RDKit_valid": ratio(
            sum(
                bool(r["raw_candidates"] and r["raw_candidates"][0]["rdkit_valid"]) for r in records
            ),
            n,
        ),
        "raw_top1_Polar_chemistry_valid": ratio(
            sum(
                bool(r["raw_candidates"] and r["raw_candidates"][0]["polar_valid"]) for r in records
            ),
            n,
        ),
        "target_Polar_formula_coverage": ratio(
            sum(
                any(c["polar_formula_match_including_H"] for c in r["raw_candidates"])
                for r in records
            ),
            n,
        ),
        "target_full_neutral_formula_coverage": ratio(
            sum(any(c["formula_match"] for c in r["raw_candidates"]) for r in records), n
        ),
        "strict_joint_formula_builder_coverage": ratio(
            sum(bool(r["accepted_unique"]) for r in records), n
        ),
        "no_qualified_candidate": ratio(sum(not r["accepted_unique"] for r in records), n),
        "legal_isomer_count": distribution([len(r["accepted_unique"]) for r in records]),
        "legal_isomer_histogram": dict(
            Counter(map(str, [len(r["accepted_unique"]) for r in records]))
        ),
        "targets_multiple_connectivity_graphs": ratio(
            sum(len(r["accepted_unique"]) > 1 for r in records), n
        ),
        "exact": {
            str(k): ratio(
                sum(any(c["polar_exact"] for c in r["accepted_unique"][:k]) for r in records), n
            )
            for k in (1, 5, 10)
        },
        "true_graph_in_strict_beam": ratio(
            sum(any(c["polar_exact"] for c in r["accepted_unique"]) for r in records), n
        ),
        "true_graph_in_raw_valid_beam": ratio(
            sum(any(c["polar_exact"] for c in r["raw_candidates"]) for r in records), n
        ),
        "true_graph_present_below_rank10": ratio(
            sum(any(c["polar_exact"] for c in r["accepted_unique"][10:]) for r in records), n
        ),
        "candidate_template_copy": ratio(
            sum(c["copy"] for r in private for c in r["candidates"]), len(strict)
        ),
        "candidate_novel_to_full_TRAIN": ratio(
            sum(c["novel"] for r in private for c in r["candidates"]), len(strict)
        ),
        "candidate_novel_with_positive_gain_vs_template_best": ratio(
            sum(c["novel"] and c["gain"] > EPS for r in private for c in r["candidates"]),
            len(strict),
        ),
    }


def paired_models(a, b):
    assert [r["row"] for r in a] == [r["row"] for r in b]
    out = {}
    for k in ("1", "5", "10"):
        delta = np.asarray([r["s"][k] for r in a]) - [r["s"][k] for r in b]
        draws = np.random.default_rng(42).integers(0, len(a), (2000, len(a)))
        out[k] = {
            **distribution(delta),
            "improved": ratio((delta > EPS).sum(), len(a)),
            "worsened": ratio((delta < -EPS).sum(), len(a)),
            "diagnostic_bootstrap_mean_95_percentile": np.quantile(
                delta[draws].mean(1), [0.025, 0.975]
            ).tolist(),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    suffix = "_partial" if args.allow_incomplete else ""
    rows = np.load(SAMPLES / "diagnostic_512.npy")
    assert row_hash(rows) == EXPECTED_COHORT
    data = PolarDataset(FEATURES, rows, split_id=1)
    meta = pq.read_table(V03 / "polar_groups.parquet").to_pydict()
    train_groups = np.unique(data.groups[data.splits == 0])
    train = frozenset(data.heads[int(g)] for g in train_groups)
    formula_frequency = Counter(tuple(data.polar_formulas[int(g)]) for g in train_groups)
    store = ShardedRowStore(V03)
    chain_rules, head_rules = load_builder_rules(V03, LIBRARY, lambda _: None)
    vocab = read_json(V03 / "chain_vocab.json")["tokens"]
    contexts = []
    for index in range(512):
        item = data[index]
        row, group = int(rows[index]), int(data.groups[rows[index]])
        target = item["polar_fragment_smiles"]
        refs = item["template_polar_smiles"]
        assert target not in train and canonical_polar(target) == target
        ts = [sim(s, target) for s in refs]
        best_index = int(np.argmax(ts))
        formula_gaps = [
            np.abs(atom_counts(parse_molecule(s)).astype(float) - data.polar_formulas[group]).sum()
            for s in refs
        ]
        count = formula_frequency[tuple(data.polar_formulas[group])]
        contexts.append(
            {
                "row": row,
                "target": target,
                "templates": refs,
                "template_rows": item["template_rows"],
                "template_top1": ts[0],
                "template_best": max(ts),
                "template_best_index": best_index,
                "best_template_formula_equal": bool(formula_gaps[best_index] == 0),
                "best_template_formula_L1": float(formula_gaps[best_index]),
                "template_count": len(refs),
                "family": meta["chemistry_family"][group],
                "length": int(meta["token_length"][group]),
                "P_count": int(data.polar_formulas[group, 4]),
                "tier": meta["polar_plausibility_tier"][group],
                "origin": meta["structure_origin"][group],
                "formula_train_graph_frequency": count,
                "formula": data.polar_formulas[group].tolist(),
                "chains": [vocab[t] for t in data.chains[row] if t],
                "full_formula": data.formulas[row].tolist(),
                "target_full": store.row(row)["canonical_full_smiles"],
            }
        )
    write(args.root / "template_context_private.json", contexts)
    all_cases = cases(rows, args.allow_incomplete)
    output, private_cases, sources = {}, {}, {}
    for key, (records, paths) in all_cases.items():
        private, errors, counters = [], Counter(), Counter()
        for record, context in zip(records, contexts):
            assert record["row"] == context["row"]
            accepted = record["accepted_unique"]
            ranked = rank_unique_accepted(record["raw_candidates"])
            assert [(c["canonical_polar"], c["decoder_score"]) for c in ranked] == [
                (c["canonical_polar"], c["decoder_score"]) for c in accepted
            ]
            assert len({c["canonical_polar"] for c in accepted}) == len(accepted)
            cs = []
            for rank, c in enumerate(accepted, 1):
                graph = c["canonical_polar"]
                # Recheck only accepted candidates with the unchanged original Builder.
                validated = score_candidate(
                    c["polar_smiles_internal"],
                    c["eos"],
                    context["chains"],
                    chain_rules,
                    head_rules,
                    context["full_formula"],
                    context["target"],
                    context["target_full"],
                    context["templates"],
                )
                assert (
                    validated["polar_valid"]
                    and validated["assembled"]
                    and validated["formula_match"]
                )
                assert validated["canonical_polar"] == graph
                assert np.array_equal(atom_counts(parse_molecule(graph)), context["formula"])
                assert validated["polar_exact"] == c["polar_exact"]
                s = sim(graph, context["target"])
                best_ref = context["templates"][context["template_best_index"]]
                info, target_info, ref_info = (
                    graph_info(graph)[1],
                    graph_info(context["target"])[1],
                    graph_info(best_ref)[1],
                )
                flags = [name for name in info if info[name] != target_info[name]]
                if not c["polar_exact"]:
                    errors.update(flags or ["unclassified"])
                shared = (
                    ref_info["phosphate_neighborhood"][0]
                    == target_info["phosphate_neighborhood"][0]
                )
                preserve = (
                    shared
                    and info["phosphate_neighborhood"][0] == ref_info["phosphate_neighborhood"][0]
                )
                modified = any(
                    info[f] != ref_info[f]
                    for f in (
                        "functional_group_graph_distances",
                        "terminal_atoms",
                        "branch_degree_patterns",
                        "ring_patterns",
                    )
                )
                counters["shared_reference_target_rooted_P_core_candidate_opportunities"] += shared
                counters["preserved_shared_rooted_P_core_candidates"] += preserve
                counters["preserved_core_modified_other_features_candidates"] += (
                    preserve and modified
                )
                counters["preserved_core_modified_positive_gain_candidates"] += (
                    preserve and modified and s > context["template_best"] + EPS
                )
                cs.append(
                    {
                        "rank": rank,
                        "graph": graph,
                        "similarity": s,
                        "gain": s - context["template_best"],
                        "copy": graph in context["templates"],
                        "novel": graph not in train,
                        "exact": c["polar_exact"],
                        "candidate_top1_ref_tanimoto": sim(graph, context["templates"][0]),
                        "candidate_best_ref_tanimoto": max(
                            sim(graph, r) for r in context["templates"]
                        ),
                        "difference_flags_vs_target": flags,
                        "preserves_shared_rooted_P_core": preserve,
                        "modifies_other_features": modified,
                    }
                )
            values = [c["similarity"] for c in cs]
            private.append(
                {
                    "row": context["row"],
                    "cohort_index": record["cohort_index"],
                    "template_top1": context["template_top1"],
                    "template_best": context["template_best"],
                    "candidate_count": len(cs),
                    "s": topn_similarities(values),
                    "s_all": max(values, default=0.0),
                    "candidates": cs,
                    "exact": {str(n): any(c["exact"] for c in cs[:n]) for n in N_VALUES},
                }
            )
        strata = defaultdict(list)
        for r, c in zip(private, contexts):
            keys = [
                "template_best:"
                + (
                    "low_<0.4"
                    if c["template_best"] < 0.4
                    else "middle_0.4_to_0.7"
                    if c["template_best"] < 0.7
                    else "high_>=0.7"
                ),
                "family:" + c["family"],
                "length:" + str(c["length"] // 16 * 16),
                "P_count:" + str(c["P_count"]),
                "tier:" + c["tier"],
                "origin:" + c["origin"],
                "formula_TRAIN_frequency:"
                + (
                    "0"
                    if c["formula_train_graph_frequency"] == 0
                    else "1_to_5"
                    if c["formula_train_graph_frequency"] <= 5
                    else "6_to_50"
                    if c["formula_train_graph_frequency"] <= 50
                    else ">50"
                ),
                "best_template_formula_equal:" + str(c["best_template_formula_equal"]),
                "best_template_formula_L1:"
                + (
                    "0"
                    if c["best_template_formula_L1"] == 0
                    else "1_to_5"
                    if c["best_template_formula_L1"] <= 5
                    else "6_to_15"
                    if c["best_template_formula_L1"] <= 15
                    else ">15"
                ),
            ]
            for label in keys:
                strata[label].append(r)
        candidates = [c for r in private for c in r["candidates"]]
        novel = [c for c in candidates if c["novel"]]
        output[key] = {
            "summary": sample_summary(private),
            "auxiliary": auxiliary(records, private),
            "strata": {k: sample_summary(v) for k, v in sorted(strata.items())},
            "candidate_similarity": distribution([c["similarity"] for c in candidates]),
            "novel_candidate_similarity": distribution([c["similarity"] for c in novel]),
            "novel_candidate_thresholds": thresholds([c["similarity"] for c in novel]),
            "candidate_to_top1_ref_similarity": distribution(
                [c["candidate_top1_ref_tanimoto"] for c in candidates]
            ),
            "candidate_to_best_ref_similarity": distribution(
                [c["candidate_best_ref_tanimoto"] for c in candidates]
            ),
            "three_way_relationship": {
                label: ratio(sum(predicate(c) for c in candidates), len(candidates))
                for label, predicate in {
                    "candidate_closer_to_target_than_any_ref": lambda c: (
                        c["similarity"] > c["candidate_best_ref_tanimoto"] + EPS
                    ),
                    "candidate_closer_to_a_ref_than_target": lambda c: (
                        c["similarity"] < c["candidate_best_ref_tanimoto"] - EPS
                    ),
                    "candidate_equal_similarity_to_target_and_best_ref": lambda c: (
                        abs(c["similarity"] - c["candidate_best_ref_tanimoto"]) <= EPS
                    ),
                }.items()
            },
            "shared_core_descriptive_proxy": dict(counters),
            "wrong_candidate_graph_difference_flags": dict(errors),
            "source_predictions_sha256": {p.as_posix(): file_hash(p) for p in paths},
        }
        private_cases[key] = private
        sources.update(output[key]["source_predictions_sha256"])
        print(
            json.dumps(
                {
                    "case": key,
                    "strict_candidates_revalidated": len(candidates),
                    "S10_mean": output[key]["summary"]["all_targets"]["10"]["similarity"]["mean"],
                    "positive_delta10": output[key]["summary"]["all_targets"]["10"][
                        "delta_vs_template_best"
                    ]["positive"],
                }
            ),
            flush=True,
        )
    paired = {}
    for beam in (20, 50):
        for a, b in (("50k_3000", "20k_3000"), ("50k_5000", "20k_3000"), ("50k_5000", "50k_3000")):
            ka, kb = f"{a}_b{beam}", f"{b}_b{beam}"
            if ka in private_cases and kb in private_cases:
                paired[f"{ka}_minus_{kb}"] = paired_models(private_cases[ka], private_cases[kb])
    report = {
        "schema": "polar_similarity_v04_v1",
        "complete": len(output) == 6,
        "cohort_rows_sha256": row_hash(rows),
        "targets": 512,
        "checkpoint_sha256": {k: file_hash(v) for k, v in CHECKPOINTS.items()},
        "template_context_private_sha256": file_hash(args.root / "template_context_private.json"),
        "template_count_histogram": dict(Counter(str(c["template_count"]) for c in contexts)),
        "cases": output,
        "paired_model_differences": paired,
        "source_predictions_sha256": sources,
        "fingerprint": {
            "boundary": "canonical Polar with attachment dummy",
            "radius": 2,
            "bits": 2048,
            "chirality": False,
            "similarity": "Tanimoto",
        },
        "no_candidate_similarity": 0,
        "comparison_tolerance": EPS,
        "ranking": "unchanged decoder-score strict canonical-graph-deduplicated order; "
        "prefix before target-similarity maximum",
        "all_accepted_candidates_Builder_formula_H_revalidated": True,
        "analysis_source_sha256": {
            p.as_posix(): file_hash(p)
            for p in (
                Path(__file__),
                Path("scripts/analyze_polar_fp50k.py"),
                Path("src/structure_pretrain/training/quality_v03.py"),
                Path("src/structure_pretrain/library_builder/assembly.py"),
                Path("src/structure_pretrain/polar_only_v04/dataset.py"),
            )
        },
        "training_started": False,
        "test_used": False,
        "diagnostic_reused_not_independent_generalization": True,
    }
    write(args.root / f"aggregate{suffix}.json", report)
    write(args.root / f"target_metrics_private{suffix}.json", private_cases)
    if not args.allow_incomplete:
        assert report["complete"]


if __name__ == "__main__":
    main()
