"""Whole-core full-validation training; fresh initialization and 8,000 TRAIN references."""

import math
import os
import sys
import time
from pathlib import Path

import torch

from structure_pretrain.core_templates.common import (
    ROOT,
    configuration,
    contract,
    digest,
    read,
    verify_data,
    write,
)
from structure_pretrain.core_templates.data import CoreData, make_model
from structure_pretrain.core_templates.model import loss as polar_loss
from structure_pretrain.formal_v2.retention import CompactCheckpoints
from structure_pretrain.formal_v2.schedule import WarmupCosine
from structure_pretrain.reported_headgroups.formal_template_data import (
    epoch_batches,
    update_selection,
)
from structure_pretrain.utils.attention import attention_backend


def optimizer_and_schedule(model, experiment):
    c = configuration()
    s = c["schedule"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=s["peak_lr"], weight_decay=c["weight_decay"]
    )
    scheduler = (
        None
        if s["kind"] == "constant"
        else WarmupCosine(
            s["peak_lr"],
            s["minimum_lr"],
            s["warmup_steps"],
            read(ROOT / "data/manifest.json")["max_optimizer_steps"],
        )
    )
    return optimizer, scheduler


def step(model, optimizer, scheduler, completed_steps, batch, pad_id, *, audit=False):
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = polar_loss(output, batch, pad_id)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite training loss; stop")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if audit:
        for name, module in model.named_children():
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            if gradients and (
                any(g is None or not bool(torch.isfinite(g).all()) for g in gradients)
                or not any(bool(torch.count_nonzero(g)) for g in gradients)
            ):
                raise ValueError("Missing/nonfinite/zero gradients in " + name)
    lr = (
        scheduler.before_optimizer_step(optimizer, completed_steps)
        if scheduler
        else optimizer.param_groups[0]["lr"]
    )
    optimizer.step()
    if scheduler:
        scheduler.after_optimizer_step()
    gold = batch["target_polar_token_ids"][:, 1:]
    mask = gold != pad_id
    return {
        "CE": float(loss.detach()),
        "gradient_norm": float(norm),
        "LR": float(lr),
        "tokens": int(mask.sum()),
        "correct": int(((output["polar_logits"].detach().argmax(-1) == gold) & mask).sum()),
    }


def payload(model, optimizer, scheduler, progress, binding, device):
    from structure_pretrain.core_templates.runtime import rng_state

    if scheduler and scheduler.updates_completed != progress["step"]:
        raise ValueError("Scheduler/global-step mismatch")
    return {
        "schema": "glycerol_core8000_full_restart",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "progress": progress,
        "contract": binding,
        "rng": rng_state(device),
    }


def restore(path, model, optimizer, scheduler, binding, device):
    from structure_pretrain.core_templates.runtime import restore_rng

    state = torch.load(path, map_location=device, weights_only=False)
    if state["contract"] != binding or state["schema"] != "glycerol_core8000_full_restart":
        raise ValueError("Whole-core checkpoint contract changed")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])
        if scheduler.updates_completed != state["progress"]["step"]:
            raise ValueError("Restored scheduler/global-step mismatch")
    elif state["scheduler"] is not None:
        raise ValueError("Constant LR checkpoint must have scheduler=None")
    restore_rng(state["rng"], device)
    return state["progress"]


def validation(model, data, event):
    model.eval()
    indices = data.indices(1)
    if len(indices) != data.manifest["split_counts"]["validation"]:
        raise ValueError("Full frozen validation required")
    indices = indices[data.lengths[indices].argsort(kind="stable")]
    total, count, correct = 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(indices), 32):
            batch = data.batch(indices[start : start + 32])
            output = model(batch)
            loss = float(polar_loss(output, batch, data.tokenizer.pad_id))
            if not math.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            gold = batch["target_polar_token_ids"][:, 1:]
            mask = gold != data.tokenizer.pad_id
            n = int(mask.sum())
            total += loss * n
            count += n
            correct += int(((output["polar_logits"].argmax(-1) == gold) & mask).sum())
            if start % 1024 == 0:
                event(
                    stage="validation_progress",
                    targets=min(start + 32, len(indices)),
                    total=len(indices),
                )
    model.train()
    return {
        "validation_token_CE": total / count,
        "validation_token_accuracy": correct / count,
        "validation_tokens": count,
        "validation_targets": len(indices),
    }


def train(experiment, out, event, resume=None):
    from structure_pretrain.core_templates.runtime import initialize

    c = configuration()
    root, out = ROOT.resolve(), Path(out).resolve()
    if out.parent != root or out.exists() or (root / "GPU_STOP.json").exists():
        raise ValueError("Fresh whole-core run and no stop latch required")
    binding = contract(root)
    for name in ("cpu_gate.json", "cuda_gate.json"):
        gate = read(root / name)
        if not gate["passed"] or gate["contract"] != binding:
            raise ValueError("Matching whole-core gates required")
    if not os.environ.get("CORE_SUPERVISOR_PID") or not torch.cuda.is_available():
        raise ValueError("Native CUDA supervisor required")
    out.mkdir()
    event(stage="verifying_immutable_data")
    verify_data(root)
    initialize(c["seed"])
    data = CoreData(root, "cuda")
    model = make_model(root).cuda()
    optimizer, scheduler = optimizer_and_schedule(model, experiment)
    progress = {
        "experiment": experiment,
        "epoch": 0,
        "batch_cursor": 0,
        "step": 0,
        "consumed_targets": 0,
        "best_validation_loss": float("inf"),
        "best_epoch": None,
        "bad_epochs": 0,
        "epoch_loss_sum": 0.0,
        "epoch_tokens": 0,
        "epoch_correct": 0,
        "validation_history": [],
    }
    if resume:
        resume = Path(resume).resolve()
        if not resume.is_relative_to(root) or resume.suffix != ".pt":
            raise ValueError("Resume only from a verified new whole-core checkpoint")
        pointer = read(resume.parent.parent / "checkpoint_retention.json")
        if (
            resume.name not in pointer["retained"]
            or digest(resume) != pointer["retained"][resume.name]["sha256"]
        ):
            raise ValueError("Unregistered or changed resume checkpoint")
        progress = restore(resume, model, optimizer, scheduler, binding, "cuda")
        if (
            progress["experiment"] != experiment
            or progress["step"]
            != progress["epoch"] * data.manifest["steps_per_epoch"] + progress["batch_cursor"]
        ):
            raise ValueError("Resume experiment/cursor inconsistent")
        if progress["epoch"] >= 80 or progress["bad_epochs"] >= 5:
            raise ValueError("Finished/early-stopped run cannot be resumed beyond its cap")
        source_best = resume.parent.parent / "best.json"
        if source_best.exists():
            best_pointer = read(source_best)
            if digest(best_pointer["path"]) != best_pointer["sha256"]:
                raise ValueError("Previous best checkpoint changed")
            write(out / "best.json", {**best_pointer, "preserved_from_resume_run": True})
    retention = CompactCheckpoints(
        out, experiment_root=root, hard_limit_bytes=c["retention"]["hard_limit_GiB"] * 2**30
    )
    import subprocess

    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write(
        out / "run_contract.json",
        {
            "contract": binding,
            "configuration": c,
            "experiment": experiment,
            "source_commit": source_commit,
            "fresh_random_initialization": not bool(resume),
            "resume_from": str(resume) if resume else None,
            "seed": c["seed"],
            "data_counts": data.manifest["split_counts"],
            "bank_members_are_prediction_targets": 8000,
            "bank_self_equivalent_references_excluded": True,
            "label_scope": c["label_scope"],
            "formula_scope": c["formula_scope"],
            "interpreter": sys.executable,
            "device": torch.cuda.get_device_name(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "retention": c["retention"],
            "precision": "FP32/Math/TF32off",
            "selected_by": "minimum complete validation Token CE",
        },
    )
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    event(
        stage="training_started",
        experiment=experiment,
        max_epochs=80,
        max_steps=data.manifest["max_optimizer_steps"],
        batch=32,
    )

    def checkpoint(label, improved=False):
        entry = retention.save(
            f"step_{progress['step']:06d}_{label}.pt",
            payload(model, optimizer, scheduler, progress, binding, "cuda"),
            best=improved,
        )
        event(
            stage="checkpoint_verified",
            step=progress["step"],
            best=improved,
            sha256=entry["sha256"],
            retained_full_checkpoints=len(read(retention.index)["retained"]),
        )

    stopped = None
    with attention_backend("math"):
        model.train()
        while progress["epoch"] < 80:
            epoch = progress["epoch"]
            batches = epoch_batches(data.indices(0), data.lengths, 32, c["seed"], epoch)
            if len(batches) != data.manifest["steps_per_epoch"]:
                raise ValueError("Unexpected complete-epoch optimizer steps")
            for cursor in range(progress["batch_cursor"], len(batches)):
                tick = time.monotonic()
                result = step(
                    model,
                    optimizer,
                    scheduler,
                    progress["step"],
                    data.batch(batches[cursor]),
                    data.tokenizer.pad_id,
                    audit=cursor == 0,
                )
                progress["step"] += 1
                progress["batch_cursor"] = cursor + 1
                progress["consumed_targets"] += len(batches[cursor])
                progress["epoch_loss_sum"] += result["CE"] * result["tokens"]
                progress["epoch_tokens"] += result["tokens"]
                progress["epoch_correct"] += result["correct"]
                if progress["step"] % 100 == 0 or cursor == 0:
                    event(
                        stage="train_progress",
                        step=progress["step"],
                        epoch=epoch + 1,
                        batch=cursor + 1,
                        training_CE=result["CE"],
                        LR=result["LR"],
                        training_token_accuracy=result["correct"] / result["tokens"],
                        gradient_norm=result["gradient_norm"],
                        seconds_per_step=time.monotonic() - tick,
                        peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                        elapsed_seconds=time.monotonic() - began,
                    )
                if progress["step"] % c["periodic_checkpoint_steps"] == 0 and cursor + 1 < len(
                    batches
                ):
                    checkpoint("periodic")
            metrics = validation(model, data, event)
            best, bad, improved = update_selection(
                progress["best_validation_loss"],
                progress["bad_epochs"],
                metrics["validation_token_CE"],
            )
            metrics.update(
                epoch=epoch + 1,
                step=progress["step"],
                LR=optimizer.param_groups[0]["lr"],
                training_token_CE=progress["epoch_loss_sum"] / progress["epoch_tokens"],
                training_token_accuracy=progress["epoch_correct"] / progress["epoch_tokens"],
            )
            progress["validation_history"].append(metrics)
            progress.update(
                epoch=epoch + 1,
                batch_cursor=0,
                best_validation_loss=best,
                bad_epochs=bad,
                epoch_loss_sum=0.0,
                epoch_tokens=0,
                epoch_correct=0,
            )
            if improved:
                progress["best_epoch"] = epoch + 1
            event(stage="validation_completed", **metrics, improved=improved, bad_epochs=bad)
            checkpoint(f"epoch_{epoch + 1:03d}", improved)
            write(out / "validation_history.json", progress["validation_history"])
            # Recheck mutable inputs every complete epoch before continuing.
            manifest = read(root / "data/manifest.json")
            if digest(root / "data/manifest.json") != binding["data_manifest_sha256"]:
                raise ValueError("Data manifest changed during training")
            for name, expected in manifest["assets_sha256"].items():
                if digest(root / "data" / name) != expected:
                    raise ValueError("Data changed during training: " + name)
            if bad >= 5:
                stopped = "patience5_complete_validation_epochs"
                break
    report = {
        "status": "completed",
        "experiment": experiment,
        "stop_reason": stopped or "80_epoch_resource_cap",
        "completed_epochs": progress["epoch"],
        "steps": progress["step"],
        "best_epoch": progress["best_epoch"],
        "best_validation_token_CE": progress["best_validation_loss"],
        "bad_epochs": progress["bad_epochs"],
        "elapsed_seconds": time.monotonic() - began,
        "cap_is_not_convergence": not bool(stopped),
        "test_used_for_selection": False,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
    }
    write(out / "training_report.json", report)
    event(stage="training_completed", **report)
    return report
