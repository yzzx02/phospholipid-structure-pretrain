"""Bounded FP50k training; immutable stage outputs, exact data/RNG resume contract."""

import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.dataset import PolarDataset, collate_polar
from structure_pretrain.polar_only_v04.fp50k_data import (
    FEATURES,
    ROOT,
    fresh_directory,
    get_contract,
)
from structure_pretrain.polar_only_v04.model import polar_loss
from structure_pretrain.polar_only_v04.prepare import SAMPLES, dump
from structure_pretrain.polar_only_v04.runtime import make_model, move, require_gpu
from structure_pretrain.utils.attention import attention_backend
from train_structure_real import ReproducibleBatchSampler, peak_ram_bytes


def validation(model, data, device, event):
    model.eval()
    total_loss, tokens, correct = 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(data), 16):
            batch = move(
                collate_polar([data[i] for i in range(start, min(start + 16, len(data)))]), device
            )
            output = model(batch)
            gold = batch["target_polar_token_ids"][:, 1:]
            mask = gold != data.tokenizer.pad_id
            n = int(mask.sum())
            total_loss += float(polar_loss(output, batch, data.tokenizer.pad_id)) * n
            correct += int(((output["polar_logits"].argmax(-1) == gold) & mask).sum())
            tokens += n
            if start % 128 == 0:
                event({"stage": "validation_progress", "targets": start})
    model.train()
    return {
        "targets": len(data),
        "tokens": tokens,
        "polar_CE": total_loss / tokens,
        "token_accuracy": correct / tokens,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    if not 1 <= args.steps <= (50 if args.probe else 5000):
        p.error("Probe cap 50; formal cap 5000")
    if args.device == "cpu" and not args.probe:
        p.error("CPU is permitted only for the regression gate")
    config, rows, contract = get_contract(args.experiment_root, verify_assets=True)
    if args.device == "cuda":
        gate = read_json(args.experiment_root / "cpu_gate.json")
        if (
            not gate["passed"]
            or gate["contract"] != contract
            or gate["interpreter"] != str(Path(sys.executable).resolve())
        ):
            raise ValueError("This interpreter's matching CPU training gate is required")
        if not args.probe:
            gpu = read_json(args.experiment_root / "gpu_training_gate.json")
            if not gpu["passed"] or gpu["contract"] != contract:
                raise ValueError("Real CUDA backward + resume gate required")
        device = require_gpu(args.experiment_root, "native_cuda")
    else:
        device = torch.device("cpu")
    out = fresh_directory(args.output, args.experiment_root)
    (out / "checkpoint").mkdir()

    def event(value):
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        if value.get("step", 0) % 20 == 0 or "stage" in value:
            print(json.dumps(value), flush=True)

    event({"stage": "loading", "device": args.device, "probe": args.probe})
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    data = PolarDataset(FEATURES, rows)
    val = PolarDataset(FEATURES, np.load(SAMPLES / "diagnostic_512.npy"), split_id=1)
    model = make_model(config, data.tokenizer, FEATURES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    step, consumed, history, val_history = 0, 0, [], []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state["contract"] != contract or state["probe"] != args.probe:
            raise ValueError("Strict checkpoint contract mismatch; probe/formal cannot mix")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, consumed = state["step"], state["consumed"]
        history, val_history = state["history"], state["validation_history"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        if args.device == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["rng_gpu"]])
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])
        del state
    if step >= args.steps or consumed != step * 16:
        raise ValueError("Invalid resume step/data consumption")
    start_step = step
    sampler = ReproducibleBatchSampler(len(data), 16, args.steps - step, 42, consumed)
    options = {"num_workers": config.train.num_workers}
    if options["num_workers"]:
        options.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(
        data,
        batch_sampler=sampler,
        collate_fn=partial(collate_polar, k=3),
        generator=torch.Generator().manual_seed(42),
        **options,
    )
    before = {
        n: torch.cat([p.detach().flatten().cpu() for p in m.parameters()])
        for n, m in model.named_children()
    }
    gradients = {}
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def save():
        state = {
            "contract": contract,
            "probe": args.probe,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "consumed": consumed,
            "history": history,
            "validation_history": val_history,
            "rng_cpu": torch.get_rng_state(),
            "rng_gpu": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        }
        dest = out / "checkpoint" / f"step_{step:06d}.pt"
        if dest.exists():
            raise FileExistsError(dest)
        torch.save(state, dest.with_suffix(".partial"))
        dest.with_suffix(".partial").replace(dest)
        event({"stage": "checkpoint_saved", "step": step, "sha256": file_hash(dest)})
        return dest

    with attention_backend("math"):
        model.train()
        for cpu in loader:
            tick = time.monotonic()
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            loss = polar_loss(model(batch), batch, data.tokenizer.pad_id)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if step == start_step or step + 1 == args.steps or (step + 1) % 1000 == 0:
                for n, m in model.named_children():
                    gs = [p.grad for p in m.parameters() if p.requires_grad]
                    if any(g is None or not torch.isfinite(g).all() for g in gs):
                        raise ValueError(f"Missing/nonfinite gradient: {n}")
                    nonzero = any(bool(torch.count_nonzero(g)) for g in gs)
                    if not nonzero:
                        raise ValueError(f"Zero gradient module: {n}")
                    gradients[n] = True
            optimizer.step()
            if args.device == "cuda":
                torch.cuda.synchronize()
            step += 1
            consumed += 16
            record = {
                "step": step,
                "polar_CE": float(loss.detach()),
                "gradient_norm_before_clip": float(norm),
                "seconds": time.monotonic() - tick,
            }
            history.append(record)
            event(record)
            if not args.probe and (step % 200 == 0 or step == args.steps):
                metrics = validation(model, val, device, event)
                val_history.append({"step": step, **metrics})
                event({"stage": "validation_completed", "step": step, **metrics})
            if step % 1000 == 0 or step == args.steps:
                checkpoint = save()
        # Reload a distinct model on a real TRAIN example, without touching saved RNG.
        model.eval()
        probe = move(collate_polar([data[0]]), device)
        with torch.inference_mode():
            logits = model(probe)["polar_logits"]
            restored = make_model(config, data.tokenizer, FEATURES).to(device).eval()
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            restored.load_state_dict(state["model"])
            torch.testing.assert_close(
                restored(probe)["polar_logits"], logits, atol=5e-5, rtol=1e-4
            )
    changes = {}
    for n, m in model.named_children():
        after = torch.cat([p.detach().flatten().cpu() for p in m.parameters()])
        delta = (after - before[n]).abs()
        changes[n] = {
            "changed_parameters": int(torch.count_nonzero(delta)),
            "total_parameters": after.numel(),
            "max_absolute_change": float(delta.max()),
        }
        if not changes[n]["changed_parameters"]:
            raise ValueError(f"Module failed to update: {n}")
    if not all(int(s["step"]) == step for s in optimizer.state.values()):
        raise ValueError("Optimizer state step mismatch")
    elapsed = time.monotonic() - started
    metrics = {
        "status": "completed",
        "probe": args.probe,
        "device": args.device,
        "step_start": start_step,
        "step": step,
        "consumed": consumed,
        "fresh_random_initialization": args.resume is None,
        "resume_checkpoint_sha256": file_hash(args.resume) if args.resume else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "module_updates": changes,
        "gradients_finite_nonzero": gradients,
        "optimizer_steps_verified": True,
        "checkpoint_reload_logits_close": True,
        "first_loss_this_stage": history[start_step]["polar_CE"],
        "last_loss": history[-1]["polar_CE"],
        "history": history,
        "validation_history": val_history,
        "elapsed_seconds": elapsed,
        "optimizer_steps_per_second": (step - start_step) / elapsed,
        "targets_per_second": (step - start_step) * 16 / elapsed,
        "peak_RAM_bytes": peak_ram_bytes(),
        "peak_VRAM_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else 0,
        "checkpoint_sha256": file_hash(checkpoint),
        "contract": contract,
        "precision": "fp32",
        "attention": "math",
        "tf32": False,
    }
    dump(out / "training_metrics.json", metrics)
    event({"stage": "completed", "step": step, "elapsed_seconds": elapsed})


if __name__ == "__main__":
    main()
