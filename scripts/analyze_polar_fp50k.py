"""Aggregate graph diagnostics; private predictions and structures never exported."""

import argparse
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.dataset import PolarDataset
from structure_pretrain.polar_only_v04.formula_beam_eval import EXPECTED_COHORT, summarize
from structure_pretrain.polar_only_v04.fp50k_data import FEATURES, ROOT
from structure_pretrain.polar_only_v04.prepare import SAMPLES, V03, row_hash
from structure_pretrain.training.quality_v03 import MORGAN

FUNCTIONS = {
    "carboxyl": "[CX3](=O)[OX2H1,O-]",
    "amine": "[NX3;!$(N-C=O)]",
    "quaternary_N": "[N+X4]",
    "hydroxyl": "[OX2H]",
    "carbonyl": "[CX3]=[OX1]",
    "nitrile": "[CX2]#[NX1]",
    "sulfonyl": "[SX4](=O)(=O)",
}
QUERIES = {key: Chem.MolFromSmarts(value) for key, value in FUNCTIONS.items()}


@lru_cache(maxsize=30000)
def graph_info(text):
    mol = parse_molecule(text)
    markers = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == 1]
    if len(markers) != 1 or markers[0].GetDegree() != 1:
        raise ValueError("Audited internal attachment required")
    root = markers[0].GetNeighbors()[0].GetIdx()

    def env(atom):
        bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, 2, atom)
        atoms = {atom}
        for b in bonds:
            bond = mol.GetBondWithIdx(b)
            atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        return Chem.MolFragmentToSmiles(
            mol, atomsToUse=sorted(atoms), bondsToUse=list(bonds), canonical=True
        )

    phosphate = (
        env(root),
        tuple(sorted(env(a.GetIdx()) for a in mol.GetAtoms() if a.GetAtomicNum() == 15)),
    )
    functional = []
    for key, query in QUERIES.items():
        matches = mol.GetSubstructMatches(query)
        # Graph distances expose moved groups as well as gained/lost groups.
        distances = sorted(
            min(len(Chem.GetShortestPath(mol, root, a)) - 1 if a != root else 0 for a in match)
            for match in matches
        )
        functional.append((key, tuple(distances)))
    terminals = tuple(
        sorted(
            (
                a.GetAtomicNum(),
                a.GetFormalCharge(),
                a.GetTotalNumHs(),
                str(a.GetBonds()[0].GetBondType()),
                a.GetNeighbors()[0].GetAtomicNum(),
            )
            for a in mol.GetAtoms()
            if a.GetAtomicNum() > 0 and a.GetDegree() == 1
        )
    )
    branches = tuple(
        sorted(
            (a.GetAtomicNum(), a.GetDegree(), a.GetIsAromatic())
            for a in mol.GetAtoms()
            if a.GetDegree() >= 3 and a.GetAtomicNum() not in (0, 15)
        )
    )
    rings = tuple(
        sorted(
            (
                len(ring),
                sum(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring),
                tuple(sorted(mol.GetAtomWithIdx(a).GetAtomicNum() for a in ring)),
            )
            for ring in mol.GetRingInfo().AtomRings()
        )
    )
    return MORGAN.GetFingerprint(mol), {
        "phosphate_neighborhood": phosphate,
        "functional_group_graph_distances": tuple(functional),
        "terminal_atoms": terminals,
        "branch_degree_patterns": branches,
        "ring_patterns": rings,
    }


def similarity(a, b):
    return float(DataStructs.TanimotoSimilarity(graph_info(a)[0], graph_info(b)[0]))


def stats(values):
    if not values:
        return {"n": 0, "mean": None, "quantiles_0_10_25_50_75_90_100": []}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "quantiles_0_10_25_50_75_90_100": np.quantile(
            values, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        ).tolist(),
    }


def analyze(records, data, head_meta):
    similarities, wrong_sim, top1_sim, best_sim, template_target, candidate_template = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    errors, strata = Counter(), defaultdict(list)
    top1_conditional = []
    copies, novel, pool_seen, true_any, true_low, multiple, novel_targets = 0, 0, 0, 0, 0, 0, 0
    train = frozenset(data.heads[int(g)] for g in np.unique(data.groups[data.splits == 0]))
    pool = np.load(ROOT / "pool/train_pool_50k.npy")
    pool_heads = frozenset(data.heads[int(g)] for g in np.unique(data.groups[pool]))
    relationships = Counter()
    for record in records:
        i = record["cohort_index"]
        item = data[i]
        row = record["row"]
        assert row == int(data.targets[i])
        target = item["polar_fragment_smiles"]
        templates = item["template_polar_smiles"]
        t_sim = similarity(target, templates[0])
        template_target.append(t_sim)
        accepted = record["accepted_unique"]
        novel_targets += target not in train
        vals = []
        for c in accepted:
            text = c["canonical_polar"]
            sim = similarity(text, target)
            vals.append(sim)
            similarities.append(sim)
            ct = similarity(text, templates[0])
            candidate_template.append(ct)
            copies += text in templates
            novel += text not in train
            pool_seen += text in pool_heads
            relationships[
                "closer_to_target"
                if sim > ct
                else "closer_to_template"
                if ct > sim
                else "equal_similarity"
            ] += 1
            if not c["polar_exact"]:
                wrong_sim.append(sim)
                a, b = graph_info(text)[1], graph_info(target)[1]
                labels = [key for key in a if a[key] != b[key]]
                for key in labels or ["unclassified"]:
                    errors[key] += 1
        # Missing strict candidates get 0 for across-all-target similarity; conditional also shown.
        top1_sim.append(vals[0] if vals else 0.0)
        if vals:
            top1_conditional.append(vals[0])
        best_sim.append(max(vals) if vals else 0.0)
        ranks = [j + 1 for j, c in enumerate(accepted) if c["polar_exact"]]
        true_any += bool(ranks)
        true_low += bool(ranks and ranks[0] > 10)
        multiple += len(accepted) > 1
        g = int(data.groups[row])
        keys = [
            "family:" + head_meta["chemistry_family"][g],
            "length:" + str(head_meta["token_length"][g] // 16 * 16),
            "template_similarity:"
            + ("<0.4" if t_sim < 0.4 else "0.4-0.7" if t_sim < 0.7 else ">=0.7"),
        ]
        for key in keys:
            strata[key].append(
                {
                    "coverage": bool(vals),
                    "exact5": any(c["polar_exact"] for c in accepted[:5]),
                    "exact10": any(c["polar_exact"] for c in accepted[:10]),
                    "top1": vals[0] if vals else 0.0,
                    "best": max(vals) if vals else 0.0,
                }
            )
    return {
        "definition": "Polar-only Morgan radius2/2048 nonchiral, internal attachment retained; "
        "no-candidate targets score zero only in explicitly all-target means",
        "accepted_polar_tanimoto": stats(similarities),
        "wrong_accepted_polar_tanimoto": stats(wrong_sim),
        "strict_top1_polar_tanimoto_all_targets": stats(top1_sim),
        "strict_top1_polar_tanimoto_among_covered": stats(top1_conditional),
        "strict_best_polar_tanimoto_all_targets": stats(best_sim),
        "target_top1_template_tanimoto": stats(template_target),
        "candidate_top1_template_tanimoto": stats(candidate_template),
        "candidate_similarity_relationship": dict(relationships),
        "exact_input_template_copy_candidates": copies,
        "accepted_novel_to_entire_train": novel,
        "accepted_seen_in_new_50k_pool": pool_seen,
        "novel_target_graphs": novel_targets,
        "graph_difference_feature_flags_wrong_candidates": dict(errors),
        "error_flag_definition": "nonexclusive graph differences, not causal error assignments; "
        "radius2 neighborhoods of rooted/all P; SMARTS group distances from "
        "attachment P; terminal atoms/bonds; non-P branch degrees; ring patterns; "
        "unclassified when none explains the difference",
        "targets_multiple_legal_connectivity_graphs": multiple,
        "true_graph_in_accepted_beam": true_any,
        "true_graph_present_below_rank10": true_low,
        "true_graph_absent_from_accepted_beam": len(records) - true_any,
        "true_graph_in_raw_valid_beam": sum(
            any(c["polar_exact"] for c in r["raw_candidates"]) for r in records
        ),
        "strata": {
            k: {
                "targets": len(v),
                "coverage": np.mean([x["coverage"] for x in v]),
                "exact5": np.mean([x["exact5"] for x in v]),
                "exact10": np.mean([x["exact10"] for x in v]),
                "top1_tanimoto_all_targets": np.mean([x["top1"] for x in v]),
                "best_tanimoto_all_targets": np.mean([x["best"] for x in v]),
            }
            for k, v in sorted(strata.items())
        },
    }


def paired_comparison(current, baseline, data):
    def values(records):
        result = []
        for record in records:
            candidates = record["accepted_unique"]
            target = data.heads[int(data.groups[record["row"]])]
            sim = [similarity(c["canonical_polar"], target) for c in candidates]
            result.append(
                [
                    bool(candidates),
                    any(c["polar_exact"] for c in candidates[:5]),
                    any(c["polar_exact"] for c in candidates[:10]),
                    sim[0] if sim else 0.0,
                    max(sim) if sim else 0.0,
                ]
            )
        return np.asarray(result, dtype=float)

    assert [r["row"] for r in current] == [r["row"] for r in baseline]
    a, b = values(current), values(baseline)
    delta = a - b
    rng = np.random.default_rng(42)
    bootstrap = delta[rng.integers(0, len(a), size=(2000, len(a)))].mean(axis=1)
    result = {
        key: {
            "mean_delta": float(delta[:, i].mean()),
            "paired_bootstrap_95_percentile": np.quantile(bootstrap[:, i], [0.025, 0.975]).tolist(),
            "targets_improved": int((delta[:, i] > 0).sum()),
            "targets_worsened": int((delta[:, i] < 0).sum()),
        }
        for i, key in enumerate(("coverage", "exact5", "exact10", "top1_tanimoto", "best_tanimoto"))
    }
    both = (a[:, 0] > 0) & (b[:, 0] > 0)
    result["both_have_candidates"] = {
        "targets": int(both.sum()),
        "top1_tanimoto_mean_delta": float(delta[both, 3].mean()) if both.any() else None,
        "best_tanimoto_mean_delta": float(delta[both, 4].mean()) if both.any() else None,
    }
    result["scope"] = (
        "descriptive paired resampling of reused diagnostic512, not independent generalization"
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, default=ROOT)
    p.add_argument("--include-baseline", action="store_true")
    p.add_argument("--baseline-only", action="store_true")
    args = p.parse_args()
    root = args.experiment_root
    rows = np.load(SAMPLES / "diagnostic_512.npy")
    assert row_hash(rows) == EXPECTED_COHORT
    data = PolarDataset(FEATURES, rows, split_id=1)
    meta = pq.read_table(V03 / "polar_groups.parquet").to_pydict()
    out = {}
    record_cache = {}
    for folder in sorted(root.iterdir()):
        if args.baseline_only:
            break
        if not (folder / "predictions_private.jsonl").exists():
            continue
        if not (folder / "host_monitor.json").exists():
            continue
        metric = read_json(folder / "metrics.json")
        host = read_json(folder / "host_monitor.json")
        assert host["stop_reason"] is None and host["protected_checkpoints_unchanged"]
        records = [
            json.loads(line)
            for line in (folder / "predictions_private.jsonl").read_text().splitlines()
        ]
        assert len(records) == metric["count"]
        key = (
            f"50k_step{metric['step']}_{metric['mode']}_beam{metric['beam_size']}"
            f"_n{metric['count']}"
        )
        out[key] = {
            "metrics": metric,
            "diagnostics": analyze(records, data, meta),
            "predictions_sha256": file_hash(folder / "predictions_private.jsonl"),
            "host_monitor_sha256": file_hash(folder / "host_monitor.json"),
        }
        record_cache[key] = records
    if args.include_baseline or args.baseline_only:
        # Only completed, disjoint jobs admitted by the previous immutable CPU revalidation.
        from report_formula_beam_cuda_v04 import aggregate_jobs

        old = Path("outputs/formula_constrained_v04_rtx4070_20261020")
        repaired = Path(str(old) + "_r1")
        cases, _ = aggregate_jobs(repaired, rows, old)
        for case in cases:
            if (
                case["encoder"] != "fingerprint"
                or case["mode"] != "elements_grammar"
                or case["beam_size"] not in (20, 50)
            ):
                continue
            records = []
            for job in case["jobs"]:
                path = (
                    Path("outputs")
                    / job["origin_experiment"]
                    / job["name"]
                    / "predictions_private.jsonl"
                )
                records.extend(json.loads(line) for line in path.read_text().splitlines())
            records.sort(key=lambda r: r["cohort_index"])
            assert [r["cohort_index"] for r in records] == list(range(512))
            key = f"20k_step3000_elements_grammar_beam{case['beam_size']}_n512"
            record_cache[key] = records
            out[key] = {
                "metrics": {
                    "strict_post_filter": summarize(records, case["beam_size"], filtered=True),
                    "raw": summarize(records, case["beam_size"], filtered=False),
                },
                "diagnostics": analyze(records, data, meta),
            }
    for key, records in record_cache.items():
        if not key.startswith("50k_") or len(records) != 512:
            continue
        metric = out[key]["metrics"]
        baseline = f"20k_step3000_elements_grammar_beam{metric['beam_size']}_n512"
        if metric["mode"] == "elements_grammar" and baseline in record_cache:
            out[key]["paired_vs_20k_3000"] = paired_comparison(
                records, record_cache[baseline], data
            )
    dest = root / ("baseline_analysis.json" if args.baseline_only else "analysis_aggregate.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {
                    "coverage": v["metrics"]["strict_post_filter"]["strict_formula_coverage"],
                    "exact": v["metrics"]["strict_post_filter"][
                        "polar_topk_exact_over_all_targets"
                    ],
                    "top1_tanimoto": v["diagnostics"]["strict_top1_polar_tanimoto_all_targets"][
                        "mean"
                    ],
                }
                for k, v in out.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
