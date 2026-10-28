"""Minimum50 epochs, patience5 thereafter; schedule horizon always80 epochs."""

import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from structure_pretrain.core_lr_study.common import (
    BASE_ROOT,
    ROOT,
    configuration,
    contract,
    digest,
    original_run,
    read,
    verify_data,
    write,
)
from structure_pretrain.core_lr_study.data import StudyData, load_shared_initialization, make_model
from structure_pretrain.core_lr_study.schedule import Schedule, stop_for_patience
from structure_pretrain.core_templates.runtime import initialize, restore_rng, rng_state
from structure_pretrain.core_templates.trainer import step, validation
from structure_pretrain.formal_v2.retention import CompactCheckpoints
from structure_pretrain.reported_headgroups.formal_template_data import (
    epoch_batches,
    update_selection,
)
from structure_pretrain.utils.attention import attention_backend

SCHEMA = "core_lr_schedule_full_restart"


def optimizer_and_schedule(model, experiment):
    c = configuration()
    s = c["experiments"][experiment]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=s["peak_lr"], weight_decay=c["weight_decay"]
    )
    scheduler = Schedule(
        s["schedule"], s["peak_lr"], s["minimum_lr"], s["warmup_steps"], c["schedule_total_steps"]
    )
    return optimizer, scheduler


def payload(model, optimizer, scheduler, progress, binding, device):
    assert scheduler.updates_completed == progress["step"]
    return {
        "schema": SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "progress": progress,
        "contract": binding,
        "rng": rng_state(device),
    }


def restore(path, model, optimizer, scheduler, binding, device):
    value = torch.load(path, map_location=device, weights_only=False)
    assert value["schema"] == SCHEMA and value["contract"] == binding
    model.load_state_dict(value["model"])
    optimizer.load_state_dict(value["optimizer"])
    scheduler.load_state_dict(value["scheduler"])
    assert scheduler.updates_completed == value["progress"]["step"]
    restore_rng(value["rng"], device)
    return copy.deepcopy(value["progress"])


def import_E3_epoch48(model, optimizer, scheduler, device):
    pointer = read(original_run() / "last.json")
    assert digest(pointer["path"]) == configuration()["original_E3_epoch48_sha256"]
    value = torch.load(pointer["path"], map_location=device, weights_only=False)
    assert value["contract"] == read(BASE_ROOT / "queue_started.json")["contract"]
    assert value["schema"] == "glycerol_core8000_full_restart"
    model.load_state_dict(value["model"])
    optimizer.load_state_dict(value["optimizer"])
    scheduler.load_state_dict(value["scheduler"])
    progress = copy.deepcopy(value["progress"])
    assert progress["epoch"] == 48 and progress["step"] == scheduler.updates_completed == 256320
    assert progress["batch_cursor"] == progress["epoch_tokens"] == 0
    progress["experiment"] = "E3"
    restore_rng(value["rng"], device)
    return progress


def initial_progress(experiment):
    return {
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


def reset_LR_accumulators(progress):
    progress.update(
        epoch_LR_sum=0.0,
        epoch_LR_count=0,
        epoch_LR_first=None,
        epoch_LR_min=float("inf"),
        epoch_LR_max=0.0,
    )


def train(experiment, out, event, resume=None):
    c, root, out = configuration(), ROOT.resolve(), Path(out).resolve()
    assert experiment in c["experiments"]
    if out.parent != root or out.exists() or (root / "GPU_STOP.json").exists():
        raise ValueError("Fresh independent LR run required; no auto-retry")
    if not os.environ.get("LR_SUPERVISOR_PID") or not torch.cuda.is_available():
        raise ValueError("Native CUDA supervision required")
    binding = contract(root)
    for name in ("cpu_gate.json", "cuda_gate.json"):
        gate = read(root / name)
        assert gate["passed"] and gate["contract"] == binding
    out.mkdir()
    event(stage="verifying_immutable_data")
    verify_data(root)
    initialize(c["seed"])
    data = StudyData(root, "cuda")
    model = make_model(root).cuda()
    optimizer, scheduler = optimizer_and_schedule(model, experiment)
    initial_sha = None
    if resume:
        resume = Path(resume).resolve()
        assert resume.is_relative_to(root) and resume.suffix == ".pt"
        index = read(resume.parent.parent / "checkpoint_retention.json")
        assert (
            resume.name in index["retained"]
            and digest(resume) == index["retained"][resume.name]["sha256"]
        )
        progress = restore(resume, model, optimizer, scheduler, binding, "cuda")
        assert progress["experiment"] == experiment
        old_best = read(resume.parent.parent / "best.json")
        assert digest(old_best["path"]) == old_best["sha256"]
        write(out / "best.json", old_best)
    elif experiment == "E3":
        progress = import_E3_epoch48(model, optimizer, scheduler, "cuda")
        write(
            out / "best.json",
            {**read(original_run() / "best.json"), "preserved_from_original_E3": True},
        )
    else:
        initial_sha = load_shared_initialization(model, root, "cuda")
        progress = initial_progress(experiment)
    assert progress["step"] == progress["epoch"] * 5340 + progress["batch_cursor"]
    assert progress["epoch"] < 80
    if "epoch_LR_count" not in progress:
        reset_LR_accumulators(progress)
    start_epoch, start_step = progress["epoch"], progress["step"]
    store = CompactCheckpoints(
        out,
        experiment_root=root,
        hard_limit_bytes=c["retention"]["owned_output_hard_limit_GiB"] * 2**30,
    )
    write(
        out / "run_contract.json",
        {
            "contract": binding,
            "configuration": c,
            "experiment": experiment,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "fresh_shared_initialization": experiment != "E3" and not bool(resume),
            "initial_parameter_sha256": initial_sha,
            "E3_independent_continuation": experiment == "E3",
            "original_E3_modified": False,
            "start_epoch": start_epoch,
            "start_step": start_step,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_targets": len(data.indices(0)),
            "validation_targets": len(data.indices(1)),
            "test_access_forbidden": True,
            "precision": "FP32/Math/TF32off",
            "interpreter": sys.executable,
            "device": torch.cuda.get_device_name(),
        },
    )
    began, stop_reason = time.monotonic(), None
    torch.cuda.reset_peak_memory_stats()
    event(
        stage="training_started",
        experiment=experiment,
        start_epoch=start_epoch,
        minimum_epochs=50,
        max_epochs=80,
        schedule_horizon_epochs=80,
        batch=32,
    )

    def checkpoint(label, improved=False):
        entry = store.save(
            f"step_{progress['step']:06d}_{label}.pt",
            payload(model, optimizer, scheduler, progress, binding, "cuda"),
            best=improved,
        )
        event(
            stage="checkpoint_verified",
            step=progress["step"],
            best=improved,
            sha256=entry["sha256"],
        )

    with attention_backend("math"):
        model.train()
        while progress["epoch"] < 80:
            epoch = progress["epoch"]
            batches = epoch_batches(data.indices(0), data.lengths, 32, c["seed"], epoch)
            assert len(batches) == 5340
            for cursor in range(progress["batch_cursor"], len(batches)):
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
                if progress["epoch_LR_first"] is None:
                    progress["epoch_LR_first"] = result["LR"]
                progress["epoch_LR_sum"] += result["LR"]
                progress["epoch_LR_count"] += 1
                progress["epoch_LR_min"] = min(progress["epoch_LR_min"], result["LR"])
                progress["epoch_LR_max"] = max(progress["epoch_LR_max"], result["LR"])
                if progress["step"] % 100 == 0 or cursor == 0:
                    event(
                        stage="train_progress",
                        experiment=experiment,
                        epoch=epoch + 1,
                        step=progress["step"],
                        training_CE=result["CE"],
                        LR=result["LR"],
                        elapsed_seconds=time.monotonic() - began,
                        peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
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
                LR_first_update=progress["epoch_LR_first"],
                LR_mean=progress["epoch_LR_sum"] / progress["epoch_LR_count"],
                LR_min=progress["epoch_LR_min"],
                LR_max=progress["epoch_LR_max"],
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
            reset_LR_accumulators(progress)
            if improved:
                progress["best_epoch"] = epoch + 1
            event(
                stage="validation_completed",
                experiment=experiment,
                **metrics,
                improved=improved,
                bad_epochs=bad,
                minimum_epoch_reached=epoch + 1 >= 50,
            )
            checkpoint(f"epoch_{epoch + 1:03d}", improved)
            write(out / "validation_history.json", progress["validation_history"])
            assert digest(root / "data/manifest.json") == binding["data_manifest_sha256"]
            assert (
                digest(BASE_ROOT / "data/manifest.json")
                == read(root / "data/manifest.json")["base_data_manifest_sha256"]
            )
            if stop_for_patience(
                epoch + 1, bad, c["minimum_epochs"], c["early_stopping_patience_epochs"]
            ):
                stop_reason = "patience5_after_minimum50_epochs"
                break
    report = {
        "status": "completed",
        "experiment": experiment,
        "stop_reason": stop_reason or "80_epoch_cap",
        "completed_epochs": progress["epoch"],
        "steps": progress["step"],
        "epochs_in_new_run": progress["epoch"] - start_epoch,
        "updates_in_new_run": progress["step"] - start_step,
        "best_epoch": progress["best_epoch"],
        "best_validation_token_CE": progress["best_validation_loss"],
        "bad_epochs": progress["bad_epochs"],
        "minimum_epochs": 50,
        "schedule_horizon_epochs": 80,
        "elapsed_seconds": time.monotonic() - began,
        "test_used_for_selection": False,
        "original_E3_unchanged": True,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
    }
    assert report["completed_epochs"] >= 50
    write(out / "training_report.json", report)
    event(stage="training_completed", **report)
    return report
