"""Frozen validation generation for independently contracted FP50k checkpoints."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.data.real_structure_dataset import ShardedRowStore
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.chemistry import derive_polar_formula
from structure_pretrain.polar_only_v04.constrained_decode import (
    constrained_beam_search,
    rank_unique_accepted,
)
from structure_pretrain.polar_only_v04.dataset import PolarDataset, collate_polar
from structure_pretrain.polar_only_v04.formula_beam_eval import (
    EXPECTED_COHORT,
    score_beams,
    summarize,
)
from structure_pretrain.polar_only_v04.fp50k_data import (
    FEATURES,
    ROOT,
    fresh_directory,
    get_contract,
)
from structure_pretrain.polar_only_v04.prepare import LIBRARY, SAMPLES, V03, dump, row_hash
from structure_pretrain.polar_only_v04.runtime import make_model, move, require_gpu
from structure_pretrain.training.quality_v03 import load_builder_rules
from structure_pretrain.utils.attention import attention_backend
from structure_pretrain.utils.resources import peak_process_memory_kib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, default=ROOT)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("raw", "elements_grammar"), default="elements_grammar")
    p.add_argument("--beam-size", type=int, choices=(1, 5, 20, 50), default=20)
    p.add_argument("--count", type=int, default=512)
    args = p.parse_args()
    if not 1 <= args.count <= 512:
        p.error("Only a bounded prefix of frozen diagnostic 512 is allowed")
    config, _, contract = get_contract(args.experiment_root)
    gate = read_json(args.experiment_root / "gpu_training_gate.json")
    if not gate["passed"] or gate["contract"] != contract:
        raise ValueError("Independent training gates required")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state["contract"] != contract or state["probe"]:
        raise ValueError("Only matching formal FP50k checkpoint allowed")
    step = state["step"]
    out = fresh_directory(args.output, args.experiment_root)
    last = 0.0

    def event(record):
        nonlocal last
        last = time.monotonic()
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def heartbeat(token):
        if time.monotonic() - last >= 2:
            event({"stage": "decoding", "token_step": token})

    event({"stage": "loading", "step": step})
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = require_gpu(args.experiment_root, "native_cuda")
    rows = np.load(SAMPLES / "diagnostic_512.npy")
    if row_hash(rows) != EXPECTED_COHORT or len(rows) != 512:
        raise ValueError("Frozen diagnostic cohort changed")
    data = PolarDataset(FEATURES, rows, split_id=1)
    model = make_model(config, data.tokenizer, FEATURES).to(device).eval()
    model.load_state_dict(state["model"])
    del state
    torch.cuda.reset_peak_memory_stats()
    chains, heads = load_builder_rules(V03, LIBRARY, event)
    vocab = read_json(V03 / "chain_vocab.json")["tokens"]
    store = ShardedRowStore(V03)
    train_heads = frozenset(data.heads[int(g)] for g in np.unique(data.groups[data.splits == 0]))
    records = []
    started = time.monotonic()
    with attention_backend("math"), torch.inference_mode():
        for index in range(args.count):
            item = data[index]
            row = int(rows[index])
            batch = move(collate_polar([item]), device)
            batch.pop("target_polar_token_ids")
            expected = derive_polar_formula(
                data.formulas[row],
                data.chains[row],
                model.known_chain_counts.cpu().numpy(),
                model.scaffold_counts.cpu().numpy(),
            )
            tick = time.monotonic()
            output = model.encode_templates(batch)
            np.testing.assert_array_equal(output["target_polar_formula"][0].cpu(), expected)
            if args.mode == "raw":
                beams = model.polar_decoder.generate(
                    output["template_memory"],
                    output["memory_padding_mask"],
                    args.beam_size,
                    progress=heartbeat,
                )[0]
            else:
                beams = constrained_beam_search(
                    model.polar_decoder,
                    output["template_memory"],
                    output["memory_padding_mask"],
                    data.tokenizer,
                    expected,
                    beam_size=args.beam_size,
                    grammar=True,
                    progress=heartbeat,
                )
            torch.cuda.synchronize()
            decode_seconds = time.monotonic() - tick
            with (out / "raw_beams_private.jsonl").open("a") as stream:
                stream.write(json.dumps({"cohort_index": index, "row": row, "beams": beams}) + "\n")
            tick = time.monotonic()
            candidates = score_beams(
                beams,
                data.tokenizer,
                polar_formula=expected,
                chain_tokens=[vocab[t] for t in data.chains[row] if t],
                chains=chains,
                heads=heads,
                full_formula=data.formulas[row],
                target_polar=item["polar_fragment_smiles"],
                target_full=store.row(row)["canonical_full_smiles"],
                template_polars=item["template_polar_smiles"],
            )
            for c in candidates:
                c["novel_to_train"] = bool(
                    c["polar_valid"] and c["canonical_polar"] not in train_heads
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
                "validation_seconds": time.monotonic() - tick,
            }
            with (out / "predictions_private.jsonl").open("a") as stream:
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
    raw = [c for r in records for c in r["raw_candidates"]]
    first = [r["raw_candidates"][0] for r in records if r["raw_candidates"]]
    metrics = {
        "status": "completed",
        "step": step,
        "pool": "50k",
        "encoder": "fingerprint",
        "mode": args.mode,
        "beam_size": args.beam_size,
        "start": 0,
        "count": args.count,
        "cohort_rows_sha256": row_hash(rows),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "raw": summarize(records, args.beam_size, filtered=False),
        "strict_post_filter": summarize(records, args.beam_size, filtered=True),
        "raw_top1_eos_rate_over_all_targets": sum(c["eos"] for c in first) / args.count,
        "raw_candidate_eos_rate": sum(c["eos"] for c in raw) / max(len(raw), 1),
        "raw_top1_length_quantiles": np.quantile(
            [len(c["token_ids"]) - 1 for c in first], [0, 0.5, 0.9, 1]
        ).tolist()
        if first
        else [],
        "elapsed_seconds": time.monotonic() - started,
        "peak_VRAM_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_RAM_kib": peak_process_memory_kib(),
        "precision": "fp32",
        "attention": "math",
        "diagnostic_reused_not_independent_generalization": True,
    }
    dump(out / "metrics.json", metrics)
    event({"stage": "completed", "seconds": metrics["elapsed_seconds"]})


if __name__ == "__main__":
    main()
