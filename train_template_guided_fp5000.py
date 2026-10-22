"""Independent bounded fixed-reference training, selected ONLY by full validation loss."""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from structure_pretrain.polar_only_v04.model import polar_loss
from structure_pretrain.reported_headgroups.formal_template_data import (
    ROOT,
    CoreData,
    config,
    contract,
    digest,
    epoch_batches,
    make_model,
    read,
    update_selection,
    verify_data,
    write,
)
from structure_pretrain.utils.attention import attention_backend


def initialize(seed=42):
    torch.set_num_threads(config()["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Preserve the historical FP32 runtime. CUDA nll_loss2d in torch2.9 does not
    # support strict deterministic_algorithms; real resume is audited numerically.
    torch.use_deterministic_algorithms(False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state(device):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"].cpu())
    if str(device).startswith("cuda"):
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def optimizer_for(model):
    c = config()
    return torch.optim.AdamW(
        model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"]
    )


def training_step(model, optimizer, batch, pad_id, audit=False):
    optimizer.zero_grad(set_to_none=True)
    loss = polar_loss(model(batch), batch, pad_id)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config()["gradient_clip"], error_if_nonfinite=True
    )
    gradients = {}
    if audit:
        for name, module in model.named_children():
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            if not grads:
                continue
            if any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
                raise ValueError("Missing/nonfinite gradient: " + name)
            if not any(bool(torch.count_nonzero(g)) for g in grads):
                raise ValueError("Zero gradient module: " + name)
            gradients[name] = True
    optimizer.step()
    n = int((batch["target_polar_token_ids"][:, 1:] != pad_id).sum())
    return float(loss.detach()), float(norm), n, gradients


def save_checkpoint(path, model, optimizer, progress, binding, device):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    value = {
        "schema": "fixed_bank_training_checkpoint_v1",
        "contract": binding,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "progress": progress,
        "rng": rng_state(device),
    }
    temporary = path.with_suffix(".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(value, temporary)
    temporary.replace(path)
    return digest(path)


def load_checkpoint(path, model, optimizer, binding, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state["contract"] != binding:
        raise ValueError("Checkpoint contract changed")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"], device)
    return state["progress"]


def validation(model, data, event):
    model.eval()
    indices = data.indices(1)
    indices = indices[np.argsort(data.lengths[indices], kind="stable")]
    total, count, correct = 0.0, 0, 0
    size = config()["validation_batch_size"]
    with torch.inference_mode():
        for start in range(0, len(indices), size):
            batch = data.batch(indices[start : start + size])
            output = model(batch)
            gold = batch["target_polar_token_ids"][:, 1:]
            mask = gold != data.tokenizer.pad_id
            n = int(mask.sum())
            loss = float(polar_loss(output, batch, data.tokenizer.pad_id))
            if not math.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            total += loss * n
            count += n
            correct += int(((output["polar_logits"].argmax(-1) == gold) & mask).sum())
            if start % 1024 == 0:
                event(
                    stage="validation_progress",
                    targets=min(start + size, len(indices)),
                    total=len(indices),
                )
    model.train()
    return {
        "validation_token_cross_entropy": total / count,
        "validation_tokens": count,
        "validation_targets": len(indices),
        "validation_token_accuracy": correct / count,
    }


def require_gates(root, binding):
    if (root / "GPU_STOP.json").exists():
        raise RuntimeError("GPU stop latched; no retry")
    cpu, gpu = read(root / "cpu_gate.json"), read(root / "gpu_training_gate.json")
    for gate in (cpu, gpu):
        if not gate["passed"] or gate["contract"] != binding:
            raise RuntimeError("Matching backward/resume gates required")
        if gate["interpreter"] != str(Path(sys.executable).resolve()):
            raise RuntimeError("Gate interpreter changed")
    if gpu["selected_runtime_sha256"] != digest(root / "selected_runtime.json"):
        raise ValueError("Benchmarked runtime changed")
    if not os.environ.get("POLAR_SUPERVISOR_PID"):
        raise RuntimeError("Formal CUDA training requires independent host supervisor")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if root != ROOT.resolve() or out.parent != root or out.exists():
        raise ValueError("Fresh run directory in independent experiment required")
    binding = contract(root)
    require_gates(root, binding)
    out.mkdir()
    (out / "checkpoint").mkdir()

    def event(**value):
        value["unix_time"] = time.time()
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    event(stage="verifying_immutable_data")
    verify_data(root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    initialize()
    selected = read(root / "selected_runtime.json")
    batch_size = selected["selected_batch_size"]
    if batch_size not in config()["batch_candidates"]:
        raise ValueError("Unbenchmarked batch")
    device = torch.device("cuda")
    data = CoreData(root, device)
    model = make_model(root).to(device)
    optimizer = optimizer_for(model)
    binding = {**binding, "selected_runtime_sha256": digest(root / "selected_runtime.json")}
    progress = {
        "epoch": 0,
        "batch_cursor": 0,
        "step": 0,
        "consumed_targets": 0,
        "batch_size": batch_size,
        "best_validation_loss": float("inf"),
        "best_epoch": None,
        "bad_epochs": 0,
        "epoch_loss_sum": 0.0,
        "epoch_tokens": 0,
        "validation_history": [],
    }
    if args.resume:
        progress = load_checkpoint(args.resume, model, optimizer, binding, device)
    if progress["batch_size"] != batch_size:
        raise ValueError("Exact resume requires same batch size")
    train = data.indices(0)
    c = config()
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    write(
        out / "run_contract.json",
        {
            "contract": binding,
            "configuration": c,
            "interpreter": sys.executable,
            "device": torch.cuda.get_device_name(),
            "batch_size": batch_size,
            "training_cores": len(train),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "fresh_random_initialization": not bool(args.resume),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "selection": "minimum full validation token cross entropy; no generation metric selection",
        },
    )
    event(
        stage="training_started",
        batch_size=batch_size,
        training_cores=len(train),
        steps_per_epoch=math.ceil(len(train) / batch_size),
        parameters=sum(p.numel() for p in model.parameters()),
    )

    def checkpoint(label, best=False):
        destination = out / "checkpoint" / f"step_{progress['step']:06d}_{label}.pt"
        sha = save_checkpoint(destination, model, optimizer, progress, binding, device)
        pointer = {
            "path": str(destination),
            "sha256": sha,
            "step": progress["step"],
            "completed_epochs": progress["epoch"],
            "best_validation_loss": progress["best_validation_loss"],
        }
        write(out / "last.json", pointer)
        if best:
            write(out / "best.json", pointer)
        event(stage="checkpoint_saved", step=progress["step"], label=label, sha256=sha, best=best)

    stopped = None
    with attention_backend("math"):
        model.train()
        while progress["epoch"] < c["max_epochs"] and progress["step"] < c["max_optimizer_steps"]:
            epoch = progress["epoch"]
            batches = epoch_batches(
                train,
                data.lengths,
                batch_size,
                c["seed"],
                epoch,
                c["length_bucket_megabatch_multiplier"],
            )
            if not 0 <= progress["batch_cursor"] <= len(batches):
                raise ValueError("Invalid resume batch cursor")
            for cursor in range(progress["batch_cursor"], len(batches)):
                tick = time.monotonic()
                batch = data.batch(batches[cursor], c["templates_per_target"])
                ce, norm, n, gradients = training_step(
                    model,
                    optimizer,
                    batch,
                    data.tokenizer.pad_id,
                    audit=progress["step"] == 0 or (progress["step"] + 1) % 1000 == 0,
                )
                progress["step"] += 1
                progress["batch_cursor"] = cursor + 1
                progress["consumed_targets"] += len(batches[cursor])
                progress["epoch_loss_sum"] += ce * n
                progress["epoch_tokens"] += n
                if progress["step"] % 20 == 0 or cursor == 0:
                    event(
                        stage="train_progress",
                        step=progress["step"],
                        epoch=epoch + 1,
                        batch=cursor + 1,
                        batches_in_epoch=len(batches),
                        training_CE=ce,
                        gradient_norm=norm,
                        seconds_per_step=time.monotonic() - tick,
                        elapsed_seconds=time.monotonic() - began,
                        peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                    )
                if progress["step"] % c["checkpoint_every_steps"] == 0 and cursor + 1 < len(
                    batches
                ):
                    checkpoint("periodic")
                if progress["step"] >= c["max_optimizer_steps"] and cursor + 1 < len(batches):
                    stopped = "local_optimizer_step_cap_before_complete_epoch"
                    checkpoint("cap")
                    break
            if stopped:
                break
            metrics = validation(model, data, event)
            best, bad, improved = update_selection(
                progress["best_validation_loss"],
                progress["bad_epochs"],
                metrics["validation_token_cross_entropy"],
            )
            metrics.update(
                epoch=epoch + 1,
                step=progress["step"],
                training_token_CE=progress["epoch_loss_sum"] / progress["epoch_tokens"],
            )
            progress["validation_history"].append(metrics)
            progress.update(
                epoch=epoch + 1,
                batch_cursor=0,
                best_validation_loss=best,
                bad_epochs=bad,
                epoch_loss_sum=0.0,
                epoch_tokens=0,
            )
            if improved:
                progress["best_epoch"] = epoch + 1
            event(stage="validation_completed", improved=improved, bad_epochs=bad, **metrics)
            checkpoint(f"epoch_{epoch + 1:03d}", best=improved)
            write(out / "validation_history.json", progress["validation_history"])
            if bad >= c["early_stopping_patience_epochs"]:
                stopped = "paper_patience5_validation_loss"
                break
    if stopped is None:
        stopped = (
            "local_max_epochs_cap"
            if progress["epoch"] >= c["max_epochs"]
            else "local_optimizer_step_cap"
        )
    report = {
        "status": "completed",
        "stop_reason": stopped,
        "step": progress["step"],
        "completed_epochs": progress["epoch"],
        "best_epoch": progress["best_epoch"],
        "best_validation_loss": progress["best_validation_loss"],
        "bad_epochs": progress["bad_epochs"],
        "batch_size": batch_size,
        "elapsed_seconds": time.monotonic() - began,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "generation_metrics_used_for_checkpoint_selection": False,
        "evaluation_pending": "generation, formula/Builder validity, Similarity@10 and functional groups",
    }
    write(out / "training_report.json", report)
    event(stage="training_completed", **report)


if __name__ == "__main__":
    main()
