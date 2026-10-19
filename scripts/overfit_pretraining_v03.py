"""Bounded real TRAIN-only memorization diagnostic before the unchanged GPU smoke."""

import argparse
import faulthandler
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from structure_pretrain.config import ExperimentConfig
from structure_pretrain.data.batching import collate_structure_examples
from structure_pretrain.data.prepare_v03 import canonical_polar, read_json
from structure_pretrain.data.real_structure_dataset import RealStructureDataset
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel
from structure_pretrain.training.losses import pretraining_loss
from structure_pretrain.utils.attention import attention_backend

THRESHOLDS = {"eos_rate": 0.95, "valid_polar_rate": 0.90, "exact_polar_rate": 0.80}


def select_rows(prepared, count, seed):
    """Distinct TRAIN heads, including the shortest/longest TRAIN polar labels."""
    if not 32 <= count <= 64:
        raise ValueError("This diagnostic requires 32..64 real targets")
    prepared = Path(prepared)
    groups = np.load(prepared / "polar_groups.npy", mmap_mode="r")
    splits = np.load(prepared / "row_splits.npy", mmap_mode="r")
    group_splits = np.load(prepared / "group_splits.npy", mmap_mode="r")
    lengths = np.asarray(
        pq.read_table(prepared / "polar_groups.parquet", columns=["token_length"])["token_length"]
    )
    eligible = np.flatnonzero(group_splits == 0)
    rng = np.random.default_rng(seed)
    selected = list(
        dict.fromkeys(
            (
                int(eligible[np.argmin(lengths[eligible])]),
                int(eligible[np.argmax(lengths[eligible])]),
            )
        )
    )
    others = np.setdiff1d(eligible, selected)
    selected.extend(rng.choice(others, count - len(selected), replace=False).tolist())
    rows = np.array(
        [int(rng.choice(np.flatnonzero((groups == g) & (splits == 0)))) for g in selected],
        dtype=np.int64,
    )
    assert len(np.unique(groups[rows])) == count and np.all(splits[rows] == 0)
    return rows


def gate_passed(metrics):
    return all(metrics[key] >= value for key, value in THRESHOLDS.items())


@torch.no_grad()
def evaluate(model, batches, tokenizer, progress=None):
    """Greedy BOS-only generation on memorized TRAIN rows; no repair or forced EOS."""
    model.eval()
    totals = dict.fromkeys(["eos", "valid", "exact", "chains", "targets"], 0)
    predictions = []
    for batch in batches:
        generated = model.generate(batch, beam_size=1)
        chains = generated["chain_logits"].argmax(-1)
        for i, candidates in enumerate(generated["polar_candidates"]):
            raw = candidates[0]
            text = tokenizer.decode(raw["ids"])
            valid, exact = False, False
            if raw["eos"]:
                try:
                    graph = canonical_polar(text)
                    valid = True
                    exact = graph == canonical_polar(batch["polar_fragment_smiles"][i])
                except ValueError:
                    pass
            totals["targets"] += 1
            totals["eos"] += int(raw["eos"])
            totals["valid"] += int(valid)
            totals["exact"] += int(exact)
            totals["chains"] += int(
                torch.equal(chains[i].sort().values, batch["target_chain_tokens"][i].sort().values)
            )
            predictions.append(
                {
                    "target_id": batch["target_id"][i],
                    "generated_polar": text,
                    "eos": raw["eos"],
                    "length_exhausted": raw["length_exhausted"],
                    "valid_polar": valid,
                    "exact_polar": exact,
                }
            )
        if progress is not None:
            progress({"stage": "generation_batch", "completed_targets": totals["targets"]})
    n = totals["targets"]
    return {
        "evaluated_targets": n,
        "eos_rate": totals["eos"] / n,
        "valid_polar_rate": totals["valid"] / n,
        "exact_polar_rate": totals["exact"] / n,
        "chain_set_exact_rate": totals["chains"] / n,
    }, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("fingerprint", "graph"), required=True)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepared", default="data/private/pretraining_v03")
    parser.add_argument("--frozen-index", default="data/private/morgan_index_v0.3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--attention-backend", choices=("default", "math"), default="default")
    parser.add_argument("--synchronize", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 5000 or args.eval_every < 1 or args.batch_size < 1:
        parser.error("Invalid bounded diagnostic configuration")
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("Working ROCm GPU is mandatory; CPU fallback is forbidden")
    preflight = read_json("outputs/pretraining_v03/rocm_preflight/preflight.json")
    migration = read_json("outputs/pretraining_v03/rocm_setup/migration_verification.json")
    if preflight["status"] != "passed" or migration["status"] != "passed":
        parser.error("Migration and GPU preflight must pass first")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostic.pid").write_text(str(os.getpid()))
    with attention_backend(args.attention_backend):
        run(args, parser, output)


def run(args, parser, output):
    trace = (output / "traceback.log").open("w")
    faulthandler.enable(file=trace)
    faulthandler.dump_traceback_later(20, file=trace)

    def log(record):
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(20, file=trace)

    log({"stage": "preparing", "attention_backend": args.attention_backend})
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(args.seed)
    rows = select_rows(args.prepared, args.targets, args.seed)
    np.save(output / "train_rows.npy", rows)
    dataset = RealStructureDataset(
        args.prepared, args.frozen_index, "train", args.encoder, "polar_excluded", rows
    )
    config = ExperimentConfig()
    config.model.encoder = args.encoder
    # Disable regularization for memorization only. Smoke retains its original defaults.
    config.model.dropout = config.model.template_dropout = 0.0
    config.model.max_polar_length = dataset.max_length
    examples = [dataset[i] for i in range(len(dataset))]
    batches = []
    for start in range(0, len(examples), args.batch_size):
        batch = collate_structure_examples(
            examples[start : start + args.batch_size], pad_id=dataset.tokenizer.pad_id
        )
        batches.append({k: v.to("cuda") if hasattr(v, "to") else v for k, v in batch.items()})
    stats = read_json(Path(args.prepared) / "formula_stats_train_only.json")
    contract = {
        "diagnostic": "TRAIN-only memorization, not validation or MS/MS accuracy",
        "model_config": asdict(config.model),
        "train_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "labels_manifest_sha256": file_hash(Path(args.prepared) / "labels_manifest.json"),
        "tokenizer_sha256": file_hash(Path(args.prepared) / "polar_tokenizer.json"),
        "formula_statistics_sha256": file_hash(
            Path(args.prepared) / "formula_stats_train_only.json"
        ),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "thresholds": THRESHOLDS,
        "retrieval_mode": "polar_excluded",
        "attention_backend": args.attention_backend,
    }
    model = StructurePretrainModel(
        config.model, len(dataset.chain_vocab), dataset.tokenizer, stats
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    start, history = 0, []
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("Overfit checkpoint contract mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, history = saved["step"], saved["history"]
        rng.bit_generator.state = saved["rng_numpy"]
        torch.set_rng_state(saved["rng_torch"])
        torch.cuda.set_rng_state_all(saved["rng_cuda"])
    if start >= args.max_steps:
        parser.error("max-steps must exceed resumed step")
    watched = {name: next(module.parameters()) for name, module in model.named_children()}
    before = {name: value.detach().clone() for name, value in watched.items()}
    started = time.monotonic()
    initial, _ = evaluate(model, batches, dataset.tokenizer, log)
    log({"stage": "initial", **initial})
    metrics, predictions = initial, []
    first_loss = None
    for step in range(start + 1, args.max_steps + 1):
        model.train()
        batch = batches[int(rng.integers(len(batches)))]
        optimizer.zero_grad(set_to_none=True)
        losses = pretraining_loss(model(batch), batch, dataset.tokenizer.pad_id, config.train)
        if args.synchronize:
            torch.cuda.synchronize()
        if not torch.isfinite(losses["loss"]):
            raise ValueError("Nonfinite overfit loss")
        losses["loss"].backward()
        if args.synchronize:
            torch.cuda.synchronize()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if args.synchronize:
            torch.cuda.synchronize()
        loss = float(losses["loss"].detach())
        if first_loss is None:
            first_loss = loss
        if step % 20 == 0 or step == 1:
            log({"step": step, "loss": loss, "gradient_norm": float(norm)})
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics, predictions = evaluate(model, batches, dataset.tokenizer, log)
            history.append({"step": step, "loss": loss, **metrics})
            passed = gate_passed(metrics)
            log({"stage": "generation", "step": step, "passed": passed, **metrics})
            state = {
                "contract": contract,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "rng_numpy": rng.bit_generator.state,
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(),
            }
            torch.save(state, output / "last.partial")
            (output / "last.partial").replace(output / "last.pt")
            report = {
                "status": "passed" if passed else "pending",
                "encoder": args.encoder,
                "real_train_targets": len(dataset),
                "optimizer_steps": step,
                "first_loss_this_run": first_loss,
                "last_loss": loss,
                "initial_generation_this_run": initial,
                "generation": metrics,
                "history": history,
                "contract": contract,
                "parameters_changed": {
                    n: not torch.equal(p, before[n]) for n, p in watched.items()
                },
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "gpu": torch.cuda.get_device_name(),
                "elapsed_seconds_this_run": time.monotonic() - started,
                "peak_VRAM_allocated_bytes": torch.cuda.max_memory_allocated(),
                "test_or_validation_targets_used": 0,
                "forced_eos_or_chemistry_repair": False,
            }
            (output / "overfit_report.json").write_text(json.dumps(report, indent=2))
            (output / "sample_predictions_private.json").write_text(
                json.dumps(predictions, indent=2)
            )
            if passed:
                if not all(report["parameters_changed"].values()):
                    raise ValueError("An overfit branch did not update")
                faulthandler.cancel_dump_traceback_later()
                return
    report["status"] = "failed"
    (output / "overfit_report.json").write_text(json.dumps(report, indent=2))
    raise SystemExit("Overfit gate failed; do not start the larger smoke")


if __name__ == "__main__":
    main()
