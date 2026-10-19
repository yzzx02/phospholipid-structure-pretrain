"""Raw greedy/beam diagnostics with externally GIVEN chains, no molecule repair."""

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from structure_pretrain.data.prepare_v03 import atom_counts, read_json
from structure_pretrain.data.real_structure_dataset import ShardedRowStore
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.chemistry import derive_polar_formula
from structure_pretrain.polar_only_v04.dataset import PolarDataset, collate_polar
from structure_pretrain.polar_only_v04.prepare import LIBRARY, SAMPLES, V03, dump, row_hash
from structure_pretrain.polar_only_v04.retrieval import quantiles, statistics_path
from structure_pretrain.polar_only_v04.runtime import load_config, make_model, move, require_gpu
from structure_pretrain.training.quality_v03 import load_builder_rules, score_candidate
from structure_pretrain.utils.attention import attention_backend


def aggregate(rows):
    n = len(rows)
    metrics = {"evaluated_targets": n}
    for key in (
        "eos",
        "rdkit_valid",
        "polar_valid",
        "polar_exact",
        "polar_formula_match",
        "formula_match",
        "full_exact",
        "template_copy",
        "assembled",
    ):
        metrics[key + "_rate"] = sum(r["raw"][key] for r in rows) / n
    metrics["top5_polar_graph_recall"] = sum(r["top5_exact"] for r in rows) / n
    metrics["top5_polar_formula_recall"] = sum(r["top5_polar_formula"] for r in rows) / n
    metrics["top5_full_formula_recall_given_chains"] = sum(r["top5_full_formula"] for r in rows) / n
    metrics["failure_categories"] = dict(Counter(r["raw"]["category"] for r in rows))
    metrics["failure_reasons"] = dict(
        Counter(r["raw"].get("reason", r["raw"]["category"]) for r in rows)
    )
    metrics["generated_length_quantiles"] = quantiles([r["generated_length"] for r in rows])
    metrics["retrieval_top1_quantiles"] = quantiles([r["retrieval_scores"][0] for r in rows])
    metrics["retrieval_topK_quantiles"] = quantiles(
        [s for r in rows for s in r["retrieval_scores"]]
    )
    metrics["target_full_Morgan_similarity_quantiles"] = quantiles(
        [r["raw"]["morgan_similarity"] for r in rows]
    )
    selected = [r["filtered"] for r in rows if r["filtered"] is not None]
    hits = sum(r["full_exact"] for r in selected)
    metrics["optional_top5_formula_filter"] = {
        "scope": "raw beams with GIVEN chains, no repair; full neutral formula acceptance",
        "selected_targets": len(selected),
        "coverage": len(selected) / n,
        "full_exact_count": hits,
        "overall_full_exact": hits / n,
        "full_exact_among_selected": hits / len(selected) if selected else None,
    }
    metrics["H_target_fraction"] = sum(r["mode"] == "H" for r in rows) / n
    metrics["fallback_targets"] = sum(r["fallback"] for r in rows)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--kind", choices=("overfit", "smoke"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attention-backend", choices=("math",), default="math")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/private/pretraining_v04"))
    args = parser.parse_args()
    config = load_config(args.config, args.kind)
    run_root = Path("outputs/pretraining_v04")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "generation_private.jsonl").exists():
        raise ValueError("Evaluation outputs cannot overwrite a previous experiment")
    (args.output / "diagnostic.pid").write_text(str(os.getpid()))

    def event(record):
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    torch.set_num_threads(config.train.cpu_threads)
    event({"stage": "loading_checkpoint"})
    device = require_gpu(run_root)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contract = state["contract"]
    if (
        contract["version"] != "v0.4"
        or contract["config"]["model"]["encoder"] != config.model.encoder
        or contract["config"] != asdict(config)
        or contract["kind"] != args.kind
    ):
        raise ValueError("Requires own v0.4 checkpoint")
    expected = {
        "source_labels_sha256": file_hash(V03 / "labels_manifest.json"),
        "tokenizer_sha256": file_hash(V03 / "polar_tokenizer.json"),
        "features_sha256": file_hash(args.root / "features_manifest.json"),
        "retrieval_sha256": file_hash(args.root / "retrieval_audit.json"),
        "formula_stats_sha256": file_hash(
            statistics_path(args.root, config.max_top_k, config.h_fraction, config.train.seed)
        ),
    }
    if any(contract[key] != value for key, value in expected.items()):
        raise ValueError("Evaluation checkpoint/data provenance mismatch")
    rows = np.load(SAMPLES / ("overfit_32.npy" if args.kind == "overfit" else "diagnostic_512.npy"))
    data = PolarDataset(
        args.root,
        rows,
        config.model.encoder,
        config.max_top_k,
        force_h=args.kind == "overfit",
        split_id=0 if args.kind == "overfit" else 1,
        h_fraction=config.h_fraction,
        seed=config.train.seed,
    )
    model = make_model(config, data.tokenizer, args.root).to(device).eval()
    model.load_state_dict(state["model"])
    checkpoint_step = state["step"]
    del state
    chains, heads = load_builder_rules(V03, LIBRARY, event)
    chain_vocab = read_json(V03 / "chain_vocab.json")["tokens"]
    chain_counts = np.load(args.root / "chain_counts.npy")
    scaffold = np.load(args.root / "scaffold_counts.npy")
    store = ShardedRowStore(V03)
    metadata = pq.read_table(V03 / "polar_groups.parquet").to_pydict()
    collate = partial(collate_polar, pad_id=data.tokenizer.pad_id, k=config.max_top_k)
    started = time.monotonic()
    with attention_backend("math"), torch.no_grad():
        tf = None
        if args.kind == "smoke":
            validation = PolarDataset(
                args.root,
                np.load(SAMPLES / "validation_pool.npy"),
                config.model.encoder,
                config.max_top_k,
                split_id=1,
                h_fraction=config.h_fraction,
                seed=config.train.seed,
            )
            loader = DataLoader(validation, batch_size=16, collate_fn=collate, num_workers=2)
            correct, total, eos_correct, eos_total, loss_sum = 0, 0, 0, 0, 0.0
            for j, cpu in enumerate(loader):
                batch = move(cpu, device)
                logits = model(batch)["polar_logits"]
                target = batch["target_polar_token_ids"][:, 1:]
                mask, eos = target != data.tokenizer.pad_id, target == data.tokenizer.eos_id
                pred = logits.argmax(-1)
                correct += int(((pred == target) & mask).sum())
                total += int(mask.sum())
                eos_correct += int(((pred == target) & eos).sum())
                eos_total += int(eos.sum())
                loss_sum += float(
                    torch.nn.functional.cross_entropy(
                        logits.transpose(1, 2),
                        target,
                        ignore_index=data.tokenizer.pad_id,
                        reduction="sum",
                    )
                )
                if j % 8 == 0:
                    event({"stage": "teacher_forcing_validation", "batches": j + 1})
            tf = {
                "targets": len(validation),
                "token_accuracy": correct / total,
                "EOS_accuracy": eos_correct / eos_total,
                "polar_CE": loss_sum / total,
            }
        records = []
        for index in range(len(data)):
            item = data[index]
            batch = move(collate([item]), device)
            raw = model.generate(batch, beam_size=1)["polar_candidates"][0][0]
            beams = model.generate(batch, beam_size=5)["polar_candidates"][0]
            row = item["row_index"]
            target_full = store.row(row)["canonical_full_smiles"]
            expected_polar = derive_polar_formula(
                np.array(data.formulas[row]), data.chains[row], chain_counts, scaffold
            )
            known_chains = [chain_vocab[t] for t in data.chains[row] if t]

            def score(candidate):
                text = data.tokenizer.decode(candidate["ids"])
                result = score_candidate(
                    text,
                    candidate["eos"],
                    known_chains,
                    chains,
                    heads,
                    data.formulas[row],
                    item["polar_fragment_smiles"],
                    target_full,
                    item["template_polar_smiles"],
                )
                result["polar_formula_match"] = bool(
                    result["polar_valid"]
                    and np.array_equal(
                        atom_counts(parse_molecule(result["canonical_polar"])), expected_polar
                    )
                )
                return result

            raw_score, beam_scores = score(raw), [score(c) for c in beams]
            g = data.groups[row]
            record = {
                "row": row,
                "raw": raw_score,
                "beams": beam_scores,
                "raw_token_ids": raw["ids"],
                "generated_length": len(raw["ids"]) - 1,
                "top5_exact": any(s["polar_exact"] for s in beam_scores),
                "top5_polar_formula": any(s["polar_formula_match"] for s in beam_scores),
                "top5_full_formula": any(s["formula_match"] for s in beam_scores),
                "filtered": next((s for s in beam_scores if s["formula_match"]), None),
                "mode": item["mode"],
                "fallback": item["fallback"],
                "retrieval_scores": item["retrieval_scores"],
                "family": metadata["chemistry_family"][g],
                "length_bin": str(metadata["token_length"][g] // 16 * 16),
                "similarity_bin": str(int(item["retrieval_scores"][0] * 5) / 5),
            }
            with (args.output / "generation_private.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            records.append(record)
            event({"stage": "raw_generation", "targets": index + 1, "total": len(data)})
    metrics = aggregate(records)
    strata = {}
    for key in ("family", "length_bin", "similarity_bin"):
        buckets = defaultdict(list)
        for record in records:
            buckets[record[key]].append(record)
        strata[key] = {name: aggregate(items) for name, items in sorted(buckets.items())}
    report = {
        "status": "completed",
        "encoder": config.model.encoder,
        "kind": args.kind,
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": file_hash(args.checkpoint),
        "cohort_rows_sha256": row_hash(rows),
        "given_chains_not_chain_prediction": True,
        "raw_no_repair_no_forced_EOS": True,
        "metrics": metrics,
        "strata": strata,
        "teacher_forcing": tf,
        "elapsed_seconds": time.monotonic() - started,
        "attention_backend": "math",
        "precision": "fp32",
    }
    dump(args.output / "evaluation_metrics.json", report)
    if args.kind == "overfit":
        passed = (
            metrics["eos_rate"] >= 0.95
            and metrics["polar_valid_rate"] >= 0.9
            and metrics["polar_exact_rate"] >= 0.8
        )
        gate = {
            "status": "passed" if passed else "failed",
            "encoder": config.model.encoder,
            "steps": checkpoint_step,
            "metrics": metrics,
            "thresholds": {"EOS": 0.95, "polar_valid": 0.9, "polar_exact": 0.8},
            "evaluation_sha256": file_hash(args.output / "evaluation_metrics.json"),
        }
        dump(args.output / "overfit_gate.json", gate)
        # Latest aggregate gate is a pointer; checkpoints and evaluations remain independent.
        dump(run_root / f"{config.model.encoder}_overfit_gate.json", gate)
    event({"stage": "completed", "evaluated_targets": len(data)})


if __name__ == "__main__":
    main()
