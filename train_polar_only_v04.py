"""Independent bounded v0.4 trainer. This entry point cannot launch full pretraining."""

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04 import SCHEMA
from structure_pretrain.polar_only_v04.dataset import PolarDataset, collate_polar
from structure_pretrain.polar_only_v04.model import polar_loss
from structure_pretrain.polar_only_v04.prepare import SAMPLES, V03, dump, row_hash
from structure_pretrain.polar_only_v04.retrieval import statistics_path
from structure_pretrain.polar_only_v04.runtime import load_config, make_model, move, require_gpu
from structure_pretrain.utils.attention import attention_backend
from train_structure_real import ReproducibleBatchSampler, peak_ram_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--kind", choices=("overfit", "smoke"), required=True)
    parser.add_argument(
        "--steps", type=int, required=True, help="Total optimizer steps, hard cap5000"
    )
    parser.add_argument("--root", type=Path, default=Path("data/private/pretraining_v04"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--attention-backend", choices=("math",), default="math")
    args = parser.parse_args()
    config = load_config(args.config, args.kind)
    limit = 500 if args.kind == "overfit" else config.train.max_steps
    if not 1 <= args.steps <= limit:
        parser.error("Bounded stage step limit exceeded")
    run_root = Path("outputs/pretraining_v04")
    if read_json(run_root / "cpu_integration_audit.json")["status"] != "passed":
        raise ValueError("Real CPU integration audit must pass before GPU training")
    if args.kind == "smoke":
        for encoder in ("fingerprint", "graph"):
            if read_json(run_root / f"{encoder}_overfit_gate.json")["status"] != "passed":
                raise ValueError("Both real 32-head overfit gates must pass before smoke")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostic.pid").write_text(str(os.getpid()))
    if (args.output / "checkpoint").exists():
        raise ValueError("Use a fresh output directory; old experiments are immutable")

    def event(record):
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    event({"stage": "constructing", "kind": args.kind, "attention": "math", "precision": "fp32"})
    device = require_gpu(run_root)
    torch.set_num_threads(config.train.cpu_threads)
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    rows = np.load(SAMPLES / ("overfit_32.npy" if args.kind == "overfit" else "train_pool.npy"))
    if len(rows) > 50000:
        raise ValueError("This bounded entry point never accepts a full-library pool")
    data = PolarDataset(
        args.root,
        rows,
        config.model.encoder,
        config.max_top_k,
        force_h=args.kind == "overfit",
        h_fraction=config.h_fraction,
        seed=config.train.seed,
    )
    if read_json(args.root / "retrieval_audit.json")["threshold"] != config.minimum_similarity:
        raise ValueError("Config threshold differs from audited retrieval")
    contract = {
        "schema": SCHEMA,
        "version": "v0.4",
        "kind": args.kind,
        "config": asdict(config),
        "outputs": ["polar_SMILES_only"],
        "known_chains_required": True,
        "target_rows_sha256": row_hash(rows),
        "train_targets": len(rows),
        "source_labels_sha256": file_hash(V03 / "labels_manifest.json"),
        "tokenizer_sha256": file_hash(V03 / "polar_tokenizer.json"),
        "features_sha256": file_hash(args.root / "features_manifest.json"),
        "retrieval_sha256": file_hash(args.root / "retrieval_audit.json"),
        "formula_stats_sha256": file_hash(
            statistics_path(args.root, config.max_top_k, config.h_fraction, config.train.seed)
        ),
        "attention_backend": "math",
        "precision": "fp32",
        "initialization": "fresh_random_v04_not_v03_weights",
    }
    dump(args.output / "resolved_config_private.json", contract)
    model = make_model(config, data.tokenizer, args.root).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    step, consumed, history = 0, 0, []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("contract") != contract:
            raise ValueError("v0.4 checkpoint/data/model mismatch; v0.3 checkpoints forbidden")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, consumed, history = state["step"], state["consumed"], state["history"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["rng_gpu"]])
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])
        del state
    if args.steps <= step:
        raise ValueError("Requested total steps must exceed resume step")
    start_step = step
    checkpoint_dir = args.output / "checkpoint"
    checkpoint_dir.mkdir()
    collate = partial(collate_polar, pad_id=data.tokenizer.pad_id, k=config.max_top_k)
    accum = config.train.gradient_accumulation
    sampler = ReproducibleBatchSampler(
        len(data), config.train.batch_size, (args.steps - step) * accum, config.train.seed, consumed
    )
    options = {"num_workers": config.train.num_workers}
    if config.train.num_workers:
        options.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(
        data,
        batch_sampler=sampler,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(config.train.seed),
        **options,
    )
    watched = {name: next(module.parameters()) for name, module in model.named_children()}
    before = {name: value.detach().clone() for name, value in watched.items()}
    gradient_ok = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def save():
        state = {
            "contract": contract,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "consumed": consumed,
            "history": history,
            "rng_cpu": torch.get_rng_state(),
            "rng_gpu": torch.cuda.get_rng_state_all(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        }
        destination = checkpoint_dir / f"step_{step:06d}.pt"
        if destination.exists():
            raise ValueError("Refusing to overwrite a retained checkpoint")
        torch.save(state, destination.with_suffix(".partial"))
        destination.with_suffix(".partial").replace(destination)
        return destination

    with attention_backend("math"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro, cpu in enumerate(loader, 1):
            batch = move(cpu, device)
            loss = polar_loss(model(batch), batch, data.tokenizer.pad_id)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite polar-only CE")
            (loss / accum).backward()
            loss_sum += float(loss.detach()) / accum
            consumed += len(cpu["target_full_formula"])
            if micro % accum:
                continue
            for name, module in model.named_children():
                grads = [p.grad for p in module.parameters() if p.requires_grad]
                if any(g is None or not torch.isfinite(g).all() for g in grads):
                    raise ValueError(f"Missing or nonfinite gradient in {name}")
                gradient_ok[name] = any(bool(g.abs().sum() > 0) for g in grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            record = {"step": step, "polar_CE": loss_sum}
            history.append(record)
            event(record)
            loss_sum = 0.0
            if step % config.train.checkpoint_every == 0:
                save()
        checkpoint = checkpoint_dir / f"step_{step:06d}.pt"
        if not checkpoint.exists():
            checkpoint = save()
        model.eval()
        probe = move(collate([data[0]]), device)
        with torch.no_grad():
            logits = model(probe)["polar_logits"]
            restored = make_model(config, data.tokenizer, args.root).to(device).eval()
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            restored.load_state_dict(saved["model"])
            actual = restored(probe)["polar_logits"]
            torch.testing.assert_close(actual, logits, atol=5e-5, rtol=1e-4)
    updates = {name: not torch.equal(value, before[name]) for name, value in watched.items()}
    if not all(updates.values()) or not all(gradient_ok.values()):
        raise ValueError("One polar-only branch failed to update")
    metrics = {
        "status": "completed",
        "kind": args.kind,
        "encoder": config.model.encoder,
        "steps_start": start_step,
        "steps_total": step,
        "consumed_samples": consumed,
        "train_targets": len(rows),
        "distinct_polar_graphs": len(np.unique(data.groups[rows])),
        "effective_batch": config.train.batch_size * accum,
        "first_loss_this_run": history[start_step]["polar_CE"],
        "last_loss": history[-1]["polar_CE"],
        "parameters_changed": updates,
        "gradients_finite_and_nonzero": gradient_ok,
        "checkpoint_reload_logits_close": True,
        "resume_optimizer_steps_verified": all(
            int(v["step"]) == step for v in optimizer.state.values()
        ),
        "real_resume_from_step": start_step if args.resume else None,
        "fresh_initialization": not bool(args.resume),
        "elapsed_seconds": time.monotonic() - started,
        "attention_backend": "math",
        "precision": "fp32",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(),
        "peak_RAM_bytes": peak_ram_bytes(),
        "peak_VRAM_allocated_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint_relative": str(checkpoint.relative_to(args.output)),
        "checkpoint_sha256": file_hash(checkpoint),
        "full_training_started": False,
    }
    dump(args.output / "training_metrics.json", metrics)
    event({"stage": "completed", "steps": step})


if __name__ == "__main__":
    main()
