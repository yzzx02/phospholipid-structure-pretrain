"""Real-model V2 CPU/CUDA backward and full-state disk-resume hard gates."""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from gate_template_guided_fp5000 import compare, cpu_copy

from structure_pretrain.formal_v2.common import ROOT, contract, digest, read, verify_data, write
from structure_pretrain.formal_v2.retention import CompactCheckpoints
from structure_pretrain.formal_v2.trainer import optimizer_and_schedule, payload, restore, step
from structure_pretrain.reported_headgroups.formal_template_data import (
    CoreData,
    epoch_batches,
    make_model,
)
from structure_pretrain.utils.attention import attention_backend
from train_template_guided_fp5000 import initialize, rng_state


def run_gate(root, out, device, experiment, event, binding):
    data = CoreData(root, device)
    batches = epoch_batches(data.indices(0), data.lengths, 32, 42, 0)[:4]
    states, losses, rates = [], [], []
    for branch in ("continuous", "restart"):
        initialize()
        model = make_model(root).to(device).train()
        optimizer, scheduler = optimizer_and_schedule(model, experiment)
        before = {
            name: torch.cat([p.detach().flatten().cpu().clone() for p in module.parameters()])
            for name, module in model.named_children()
            if any(True for _ in module.parameters())
        }
        branch_losses, branch_rates = [], []
        for i, indices in enumerate(batches):
            value = step(
                model,
                optimizer,
                scheduler,
                i,
                data.batch(indices),
                data.tokenizer.pad_id,
                audit=True,
            )
            branch_losses.append(value["CE"])
            branch_rates.append(value["LR"])
            event(
                stage="resume_gate_progress",
                experiment=experiment,
                device=device,
                branch=branch,
                step=i + 1,
                loss=value["CE"],
                LR=value["LR"],
            )
            if branch == "restart" and i == 1:
                progress = {
                    "epoch": 0,
                    "step": 2,
                    "batch_cursor": 2,
                    "best_validation_loss": 1.5,
                    "best_epoch": 1,
                    "bad_epochs": 3,
                    "validation_history": [{"epoch": 1, "CE": 1.5}],
                }
                store = CompactCheckpoints(out / (experiment + "_resume"))
                entry = store.save(
                    "step_000002_periodic.pt",
                    payload(model, optimizer, scheduler, progress, binding, device),
                )
                del model, optimizer, scheduler
                gc.collect()
                model = make_model(root).to(device).train()
                optimizer, scheduler = optimizer_and_schedule(model, experiment)
                recovered = restore(entry["path"], model, optimizer, scheduler, binding, device)
                assert recovered == progress
                rebuilt = epoch_batches(data.indices(0), data.lengths, 32, 42, recovered["epoch"])
                np.testing.assert_array_equal(rebuilt[recovered["batch_cursor"]], batches[2])
                # The real model/optimizer checkpoint is a temporary gate asset.
                # Preserve its hash and verification log, then remove this owned file.
                checkpoint = Path(entry["path"])
                assert (
                    checkpoint.parent == store.folder.resolve()
                    and digest(checkpoint) == entry["sha256"]
                )
                checkpoint.unlink()
                write(
                    store.run / "gate_checkpoint_consumed.json",
                    {
                        **entry,
                        "removed_after_successful_restore": True,
                        "no_training_weights_deleted": True,
                    },
                )
        changed = {}
        for name, module in model.named_children():
            if name in before:
                after = torch.cat([p.detach().flatten().cpu() for p in module.parameters()])
                changed[name] = bool(torch.count_nonzero(after - before[name]))
        assert all(changed.values())
        model.eval()
        with torch.inference_mode():
            logits = model(data.batch(batches[0]))["polar_logits"]
        states.append(
            {
                "model": cpu_copy(model.state_dict()),
                "optimizer": cpu_copy(optimizer.state_dict()),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "logits": cpu_copy(logits),
                "rng": cpu_copy(rng_state(device)),
            }
        )
        losses.append(branch_losses)
        rates.append(branch_rates)
        del model, optimizer, scheduler, logits
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    tolerance = 0.0 if device == "cpu" else 1e-5
    difference = compare(states[0], states[1], tolerance)
    assert rates[0] == rates[1]
    assert max(abs(a - b) for a, b in zip(*losses)) <= tolerance
    # A longest-label batch must also complete one real backward update.
    initialize()
    model = make_model(root).to(device).train()
    optimizer, scheduler = optimizer_and_schedule(model, experiment)
    longest = data.indices(0)[np.argsort(data.lengths[data.indices(0)])[-32:]]
    value = step(
        model, optimizer, scheduler, 0, data.batch(longest), data.tokenizer.pad_id, audit=True
    )
    return {
        "experiment": experiment,
        "steps": 4,
        "restart_after_step": 2,
        "batch_size": 32,
        "real_disk_reload": True,
        "maximum_model_optimizer_scheduler_logits_difference": difference,
        "tolerance": tolerance,
        "losses": losses,
        "actual_LRs": rates,
        "same_next_batch_rng_and_patience": True,
        "all_modules_updated": changed,
        "longest_label_backward_passed": True,
        "longest_batch_CE": value["CE"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = ROOT.resolve(), args.output.resolve()
    if out.parent != root or out.exists():
        raise ValueError("Fresh independent V2 gate output required")
    binding = contract(root)
    if args.device == "cuda":
        gate = read(root / "cpu_gate.json")
        if (
            not gate["passed"]
            or gate["contract"] != binding
            or not os.environ.get("POLAR_SUPERVISOR_PID")
        ):
            raise ValueError("Matching CPU gate and supervisor required")
        if (root / "GPU_STOP.json").exists() or not torch.cuda.is_available():
            raise ValueError("CUDA unavailable or stop latched; no retry")
    out.mkdir()

    def event(**row):
        row["unix_time"] = time.time()
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    event(stage="verifying_immutable_data", device=args.device)
    began = time.monotonic()
    verify_data(root)
    with attention_backend("math"):
        results = [
            run_gate(root, out, args.device, experiment, event, binding)
            for experiment in ("E1", "E2", "E3")
        ]
    report = {
        "passed": True,
        "contract": binding,
        "device": args.device,
        "interpreter": str(Path(sys.executable).resolve()),
        "checks": results,
        "elapsed_seconds": time.monotonic() - began,
    }
    write(out / "gate_report.json", report)
    write(root / ("cpu_gate.json" if args.device == "cpu" else "cuda_gate.json"), report)
    event(stage="gate_passed", device=args.device, elapsed_seconds=report["elapsed_seconds"])


if __name__ == "__main__":
    main()
