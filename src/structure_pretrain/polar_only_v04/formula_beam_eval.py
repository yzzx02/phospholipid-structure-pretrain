"""Bounded inference-only comparison on the frozen 512 validation heads."""

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from structure_pretrain.data.prepare_v03 import atom_counts, read_json
from structure_pretrain.data.real_structure_dataset import ShardedRowStore
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.chemistry import derive_polar_formula
from structure_pretrain.polar_only_v04.constrained_decode import (
    ElementBudget,
    SyntaxState,
    constrained_beam_search,
    rank_unique_accepted,
    syntax_step,
)
from structure_pretrain.polar_only_v04.dataset import PolarDataset, collate_polar
from structure_pretrain.polar_only_v04.prepare import LIBRARY, SAMPLES, V03, dump, row_hash
from structure_pretrain.polar_only_v04.retrieval import statistics_path
from structure_pretrain.polar_only_v04.runtime import load_config, make_model, move, require_gpu
from structure_pretrain.training.quality_v03 import load_builder_rules, score_candidate
from structure_pretrain.utils.attention import attention_backend
from structure_pretrain.utils.resources import peak_process_memory_kib

EXPERIMENT = Path("outputs/formula_constrained_v04_20261020")
EXPECTED_COHORT = "694d0d4c0c12330f4c555724c63dd8eeb026ff499f332108e6d4f6aae87c4fcd"
MODES = ("raw", "elements", "elements_grammar")


def score_beams(
    beams,
    tokenizer,
    *,
    polar_formula,
    chain_tokens,
    chains,
    heads,
    full_formula,
    target_polar,
    target_full,
    template_polars,
):
    """Truth is used for metrics only; acceptance uses supplied formulas/chains."""
    results = []
    for rank, candidate in enumerate(beams, 1):
        text = tokenizer.decode(candidate["ids"])
        scored = score_candidate(
            text,
            candidate["eos"],
            chain_tokens,
            chains,
            heads,
            full_formula,
            target_polar,
            target_full,
            template_polars,
        )
        polar_match = bool(
            scored["polar_valid"]
            and np.array_equal(
                atom_counts(parse_molecule(scored["canonical_polar"])), polar_formula
            )
        )
        results.append(
            {
                **scored,
                "source_rank": rank,
                "token_ids": candidate["ids"],
                "polar_smiles_internal": text,
                "log_probability": candidate["log_probability"],
                "decoder_score": candidate["log_probability"] / (len(candidate["ids"]) - 1),
                "length_exhausted": candidate["length_exhausted"],
                "polar_formula_match_including_H": polar_match,
                "accepted": bool(
                    candidate["eos"]
                    and polar_match
                    and scored["assembled"]
                    and scored["formula_match"]
                ),
            }
        )
    return results


def summarize(records, beam_size, *, filtered):
    n = len(records)
    if not n:
        raise ValueError("Cannot aggregate an empty cohort")
    raw = [c for r in records for c in r["raw_candidates"]]
    lists = [r["accepted_unique"] if filtered else r["raw_candidates"] for r in records]
    counts = [len(r["accepted_unique"]) for r in records]
    accepted = [c for r in records for c in r["accepted_unique"]]
    exact = {
        str(k): (
            sum(any(c["polar_exact"] for c in cs[:k]) for cs in lists) / n
            if beam_size >= k
            else None
        )
        for k in (1, 5, 10)
    }
    first = [cs[0] for cs in lists if cs]
    selected_formula = [c for c in first if c["accepted"]]
    return {
        "targets": n,
        "returned_candidates": sum(len(cs) for cs in lists),
        "polar_topk_exact_over_all_targets": exact,
        "raw_candidate_count": len(raw),
        "raw_chemical_valid_polar_fraction": sum(c["polar_valid"] for c in raw) / len(raw)
        if raw
        else 0.0,
        "returned_chemical_valid_polar_fraction": sum(c["polar_valid"] for cs in lists for c in cs)
        / max(sum(map(len, lists)), 1),
        "raw_top1_chemical_valid_polar_rate": sum(
            bool(r["raw_candidates"] and r["raw_candidates"][0]["polar_valid"]) for r in records
        )
        / n,
        "strict_formula_coverage": sum(c > 0 for c in counts) / n,
        "no_qualified_candidate_rate": sum(c == 0 for c in counts) / n,
        "accepted_unique_candidates": len(accepted),
        "accepted_unique_per_target": {
            "mean": sum(counts) / n,
            "min": min(counts),
            "max": max(counts),
            "median": float(np.median(counts)),
            "histogram": dict(sorted(Counter(counts).items())),
        },
        "formula_correct_wrong_polar_structure_fraction_among_accepted": sum(
            not c["polar_exact"] for c in accepted
        )
        / len(accepted)
        if accepted
        else None,
        "formula_correct_wrong_top1_fraction_among_formula_correct_top1": sum(
            not c["polar_exact"] for c in selected_formula
        )
        / len(selected_formula)
        if selected_formula
        else None,
        "formula_correct_wrong_top1_rate_over_all_targets": sum(
            not c["polar_exact"] for c in selected_formula
        )
        / n,
        "accepted_novel_to_train_candidates": sum(c["novel_to_train"] for c in accepted),
        "raw_failure_categories": dict(Counter(c.get("category", "valid") for c in raw)),
        "raw_failure_reasons": dict(
            Counter(c.get("reason", c.get("category", "valid")) for c in raw)
        ),
        "decode_seconds": sum(r["decode_seconds"] for r in records),
        "validation_seconds": sum(r["validation_seconds"] for r in records),
    }


def provenance(state, config, root, checkpoint):
    contract = state["contract"]
    if (
        state["step"] != 3000
        or contract["version"] != "v0.4"
        or contract["kind"] != "smoke"
        or contract["config"] != asdict(config)
    ):
        raise ValueError("Requires matching frozen v0.4 3000-step smoke checkpoint")
    checks = {
        "source_labels_sha256": file_hash(V03 / "labels_manifest.json"),
        "tokenizer_sha256": file_hash(V03 / "polar_tokenizer.json"),
        "features_sha256": file_hash(root / "features_manifest.json"),
        "retrieval_sha256": file_hash(root / "retrieval_audit.json"),
        "formula_stats_sha256": file_hash(
            statistics_path(root, config.max_top_k, config.h_fraction, config.train.seed)
        ),
    }
    if any(contract[k] != v for k, v in checks.items()):
        raise ValueError("Checkpoint/data/tokenizer provenance mismatch")
    return {**checks, "checkpoint_sha256": file_hash(checkpoint), "step": 3000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("fingerprint", "graph"), required=True)
    parser.add_argument("--mode", choices=MODES, default="elements_grammar")
    parser.add_argument("--beam-size", type=int, choices=(2, 5, 20, 50, 128), default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT)
    parser.add_argument("--gpu-backend", choices=("rocm", "native_cuda"), default="rocm")
    args = parser.parse_args()
    experiment = args.experiment_root
    protected = (EXPERIMENT.resolve(), Path("outputs/pretraining_v04").resolve())
    if not 0 <= args.start < 512 or not 1 <= args.count <= 512 - args.start:
        parser.error("Use a bounded slice of the frozen 512-head cohort")
    if args.gpu_backend == "native_cuda" and (
        not experiment.resolve().is_relative_to(Path("outputs").resolve())
        or any(
            experiment.resolve() == p
            or experiment.resolve().is_relative_to(p)
            or p.is_relative_to(experiment.resolve())
            for p in protected
        )
    ):
        parser.error("Native CUDA requires its own experiment root under outputs/")
    if not args.output.resolve().is_relative_to(experiment.resolve()):
        parser.error("Separate formula-experiment output root is mandatory")
    if args.output.exists():
        parser.error("Fresh output directory required; no checkpoint/report overwrites")
    args.output.mkdir(parents=True)
    (args.output / "diagnostic.pid").write_text(str(os.getpid()))
    last_heartbeat = 0.0

    def event(record):
        nonlocal last_heartbeat
        last_heartbeat = time.monotonic()
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def heartbeat(step):
        if time.monotonic() - last_heartbeat > 2:
            event({"stage": "decoding", "token_step": step})

    torch.set_num_threads(4)
    if args.gpu_backend == "native_cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    event({"stage": "loading", "device": args.device})
    if args.device == "cuda":
        gate = read_json(experiment / "cpu_gate.json")
        if not gate.get("passed"):
            raise ValueError("This experiment's CPU gate must pass before GPU inference")
        if args.gpu_backend == "native_cuda" and (
            gate.get("gpu_backend") != "native_cuda"
            or gate.get("interpreter") != str(Path(sys.executable).resolve())
        ):
            raise ValueError("Native CUDA requires this interpreter's independent CPU gate")
        device = require_gpu(experiment, args.gpu_backend)
        if args.gpu_backend == "rocm":
            require_gpu(Path("outputs/pretraining_v04"))
    else:
        device = torch.device("cpu")
    key = "fp" if args.encoder == "fingerprint" else "gine"
    root = Path("data/private/pretraining_v04")
    config = load_config(f"configs/polar_only_v04_{key}.yaml")
    checkpoint = Path(f"outputs/pretraining_v04/runs/{key}_smoke_3000/checkpoint/step_003000.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    inputs = provenance(state, config, root, checkpoint)
    rows = np.load(SAMPLES / "diagnostic_512.npy")
    if len(rows) != 512 or row_hash(rows) != EXPECTED_COHORT:
        raise ValueError("Do not change the fixed validation cohort")
    # Instantiate all512 before slicing: retrieval-mode allocation stays identical.
    data = PolarDataset(
        root,
        rows,
        args.encoder,
        config.max_top_k,
        split_id=1,
        h_fraction=config.h_fraction,
        seed=config.train.seed,
    )
    if len(np.unique(data.groups[rows])) != 512:
        raise ValueError("Validation heads must be distinct")
    model = make_model(config, data.tokenizer, root).to(device).eval()
    model.load_state_dict(state["model"])
    del state
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    chain_counts = np.load(root / "chain_counts.npy")
    scaffold = np.load(root / "scaffold_counts.npy")
    if args.audit_only:
        if args.device != "cpu":
            raise ValueError("Audit stage is CPU only")
        errors = []
        for index in range(512):
            row = int(rows[index])
            text = data.heads[data.groups[row]]
            expected = derive_polar_formula(
                data.formulas[row], data.chains[row], chain_counts, scaffold
            )
            if not np.array_equal(atom_counts(parse_molecule(text)), expected):
                errors.append({"index": index, "reason": "target_formula"})
            budget = ElementBudget(data.tokenizer, expected)
            consumed, syntax = np.zeros(9, dtype=np.int64), SyntaxState()
            for token in data.tokenizer.encode(text)[1:]:
                if not budget.allowed(consumed, syntax)[token]:
                    errors.append(
                        {
                            "index": index,
                            "reason": "reference_prefix_rejected",
                            "token": data.tokenizer.tokens[token],
                        }
                    )
                    break
                consumed += budget.increments[token]
                if token != data.tokenizer.eos_id:
                    syntax = syntax_step(syntax, data.tokenizer.tokens[token])
        batch = move(collate_polar([data[0]], k=config.max_top_k), device)
        with attention_backend("math"), torch.inference_mode():
            output = model.encode_templates(
                {k: v for k, v in batch.items() if k != "target_polar_token_ids"}
            )
            for grammar in (False, True):
                candidates = constrained_beam_search(
                    model.polar_decoder,
                    output["template_memory"],
                    output["memory_padding_mask"],
                    data.tokenizer,
                    output["target_polar_formula"][0].numpy(),
                    beam_size=2,
                    max_length=12,
                    grammar=grammar,
                )
                if any(not np.isfinite(c["log_probability"]) for c in candidates):
                    errors.append({"reason": "nonfinite_CPU_decode"})
        audit = {
            "passed": not errors,
            "encoder": args.encoder,
            "checked_targets": 512,
            "real_decoder_prefix_modes": 2,
            "errors": errors,
            "inputs": inputs,
            "device": "cpu",
            "cohort_rows_sha256": row_hash(rows),
        }
        dump(args.output / "cpu_audit.json", audit)
        event({"stage": "CPU_audit_completed", "passed": audit["passed"], "errors": len(errors)})
        if errors:
            raise ValueError("CPU formula/grammar integration audit failed")
        return
    chains, heads = load_builder_rules(V03, LIBRARY, event)
    chain_vocab = read_json(V03 / "chain_vocab.json")["tokens"]
    store = ShardedRowStore(V03)
    train_heads = frozenset(data.heads[int(g)] for g in np.unique(data.groups[data.splits == 0]))
    records = []
    started = time.monotonic()
    with attention_backend("math"), torch.inference_mode():
        for index in range(args.start, args.start + args.count):
            item = data[index]
            batch = move(collate_polar([item], k=config.max_top_k), device)
            batch.pop("target_polar_token_ids")  # never pass truth labels to generation
            row = int(rows[index])
            expected = derive_polar_formula(
                data.formulas[row], data.chains[row], chain_counts, scaffold
            )
            decode_start = time.monotonic()
            output = model.encode_templates(batch)
            np.testing.assert_array_equal(output["target_polar_formula"][0].cpu().numpy(), expected)
            if args.mode == "raw":
                beams = model.polar_decoder.generate(
                    output["template_memory"],
                    output["memory_padding_mask"],
                    args.beam_size,
                    progress=heartbeat,
                )
                beams = beams[0]
            else:
                beams = constrained_beam_search(
                    model.polar_decoder,
                    output["template_memory"],
                    output["memory_padding_mask"],
                    data.tokenizer,
                    expected,
                    beam_size=args.beam_size,
                    grammar=args.mode == "elements_grammar",
                    progress=heartbeat,
                )
            if args.device == "cuda":
                torch.cuda.synchronize()
            decode_seconds = time.monotonic() - decode_start
            with (args.output / "raw_beams_private.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "cohort_index": index,
                            "row": row,
                            "beams": beams,
                        }
                    )
                    + "\n"
                )
            validation_start = time.monotonic()
            candidates = score_beams(
                beams,
                data.tokenizer,
                polar_formula=expected,
                chain_tokens=[chain_vocab[t] for t in data.chains[row] if t],
                chains=chains,
                heads=heads,
                full_formula=data.formulas[row],
                target_polar=item["polar_fragment_smiles"],
                target_full=store.row(row)["canonical_full_smiles"],
                template_polars=item["template_polar_smiles"],
            )
            for candidate in candidates:
                candidate["novel_to_train"] = bool(
                    candidate["polar_valid"] and candidate["canonical_polar"] not in train_heads
                )
            selected = rank_unique_accepted(candidates)
            record = {
                "cohort_index": index,
                "row": row,
                "mode": item["mode"],
                "raw_candidates": candidates,
                "accepted_unique": selected,
                "failure": None if selected else "no_qualified_candidate",
                "decode_seconds": decode_seconds,
                "validation_seconds": time.monotonic() - validation_start,
            }
            with (args.output / "predictions_private.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            records.append(record)
            event(
                {
                    "stage": "target_completed",
                    "targets": len(records),
                    "total": args.count,
                    "accepted_unique": len(selected),
                }
            )
    metrics = {
        "status": "completed",
        "encoder": args.encoder,
        "mode": args.mode,
        "beam_size": args.beam_size,
        "start": args.start,
        "count": args.count,
        "cohort_rows_sha256": row_hash(rows),
        "inputs": inputs,
        "raw": summarize(records, args.beam_size, filtered=False),
        "strict_post_filter": summarize(records, args.beam_size, filtered=True),
        "elapsed_seconds": time.monotonic() - started,
        "peak_VRAM_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else 0,
        "peak_RAM_kib": peak_process_memory_kib(),
        "device": args.device,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "gpu_backend": args.gpu_backend,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_enabled": torch.backends.cudnn.allow_tf32,
        "attention_backend": "math",
        "precision": "fp32",
        "training_started": False,
    }
    dump(args.output / "metrics.json", metrics)
    event({"stage": "completed", "elapsed_seconds": metrics["elapsed_seconds"]})


if __name__ == "__main__":
    main()
