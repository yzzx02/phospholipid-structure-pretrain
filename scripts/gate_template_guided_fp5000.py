"""Real-data CPU/CUDA backward, exact restart and bounded batch16/32 speed gates."""

import argparse
import copy
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structure_pretrain.reported_headgroups.formal_template_data import (
    ROOT,
    CoreData,
    config,
    contract,
    digest,
    epoch_batches,
    make_model,
    read,
    verify_data,
    write,
)
from structure_pretrain.utils.attention import attention_backend
from train_template_guided_fp5000 import (
    initialize,
    load_checkpoint,
    optimizer_for,
    rng_state,
    save_checkpoint,
    training_step,
)


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def compare(a, b, tolerance):
    if isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
        return 0.0
    if torch.is_tensor(a):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError("Restart tensor shape/dtype mismatch")
        maximum = float((a - b).abs().max()) if a.numel() else 0.0
        if maximum > tolerance:
            raise AssertionError(f"Restart mismatch: {maximum} > {tolerance}")
        return maximum
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((compare(a[k], b[k], tolerance) for k in a), default=0.0)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max((compare(x, y, tolerance) for x, y in zip(a, b)), default=0.0)
    assert a == b
    return 0.0


def backward_resume(root, out, device, batch_size, event, binding):
    data = CoreData(root, device)
    batches = epoch_batches(data.indices(0), data.lengths, batch_size, 42, 0)
    batches = batches[:4]
    baseline, restarted = None, None
    changed, gradients = {}, {}
    losses = []
    for branch in ("continuous", "restart"):
        initialize()
        model = make_model(root).to(device)
        optimizer = optimizer_for(model)
        before = {
            n: torch.cat([p.detach().flatten().cpu().clone() for p in m.parameters()])
            for n, m in model.named_children()
            if any(True for _ in m.parameters())
        }
        branch_losses = []
        model.train()
        for step, indices in enumerate(batches):
            ce, norm, n, audit = training_step(
                model, optimizer, data.batch(indices), data.tokenizer.pad_id, audit=True
            )
            gradients.update(audit)
            branch_losses.append(ce)
            event(
                stage="resume_gate_progress",
                device=str(device),
                branch=branch,
                step=step + 1,
                batch_size=batch_size,
                loss=ce,
            )
            if branch == "restart" and step == 1:
                path = out / f"restart_{str(device)}_batch{batch_size}.pt"
                expected_progress = {
                    "epoch": 0,
                    "step": 2,
                    "batch_cursor": 2,
                    "consumed_targets": 2 * batch_size,
                    "best_validation_loss": 2.23,
                    "bad_epochs": 3,
                    "best_epoch": 1,
                    "validation_history": [{"epoch": 1, "validation_token_cross_entropy": 2.23}],
                }
                save_checkpoint(path, model, optimizer, expected_progress, binding, device)
                del model, optimizer
                gc.collect()
                model = make_model(root).to(device)
                optimizer = optimizer_for(model)
                progress = load_checkpoint(path, model, optimizer, binding, device)
                assert progress == expected_progress
                rebuilt = epoch_batches(
                    data.indices(0), data.lengths, batch_size, 42, progress["epoch"]
                )
                np.testing.assert_array_equal(rebuilt[progress["batch_cursor"]], batches[2])
                model.train()
        for name, module in model.named_children():
            if name in before:
                after = torch.cat([p.detach().flatten().cpu() for p in module.parameters()])
                changed[name] = bool(torch.count_nonzero(after - before[name]))
        assert all(changed.values()), "Every learned module must update"
        model.eval()
        with torch.inference_mode():
            logits = model(data.batch(batches[0]))["polar_logits"]
        state = {
            "model": cpu_copy(model.state_dict()),
            "optimizer": cpu_copy(optimizer.state_dict()),
            "logits": cpu_copy(logits),
            "rng_cpu": torch.get_rng_state().clone(),
            "rng_cuda": cpu_copy(rng_state(device)["cuda"]),
            "rng_numpy": copy.deepcopy(np.random.get_state()),
            "rng_python": random.getstate(),
        }
        if branch == "continuous":
            baseline = state
        else:
            restarted = state
        losses.append(branch_losses)
        del model, optimizer, logits, state
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    tolerance = 0.0 if str(device) == "cpu" else 1e-5
    maximum = compare(baseline, restarted, tolerance)
    assert max(abs(a - b) for a, b in zip(*losses)) <= tolerance
    return {
        "batch_size": batch_size,
        "steps": 4,
        "restart_after_step": 2,
        "maximum_model_optimizer_logits_difference": maximum,
        "absolute_tolerance": tolerance,
        "losses": losses,
        "all_module_gradients_finite_nonzero": gradients,
        "all_modules_updated": changed,
        "real_disk_checkpoint_reload": True,
        "same_next_batches_and_rng": True,
    }


def benchmark(root, out, event):
    data = CoreData(root, "cuda")
    measurements = []
    for size in config()["batch_candidates"]:
        initialize()
        model = make_model(root).cuda()
        optimizer = optimizer_for(model)
        model.train()
        batches = epoch_batches(data.indices(0), data.lengths, size, 42, 0)
        torch.cuda.reset_peak_memory_stats()
        # Includes a real maximum-length batch before timing; no OOM fallback/retry.
        long_ids = data.indices(0)[np.argsort(data.lengths[data.indices(0)])[-size:]]
        training_step(model, optimizer, data.batch(long_ids), data.tokenizer.pad_id, audit=True)
        for i in range(6):
            training_step(model, optimizer, data.batch(batches[i]), data.tokenizer.pad_id)
        torch.cuda.synchronize()
        began = time.perf_counter()
        targets = 0
        for i in range(64):
            ce, _, _, _ = training_step(
                model, optimizer, data.batch(batches[i + 6]), data.tokenizer.pad_id
            )
            targets += len(batches[i + 6])
            if i % 16 == 0:
                event(stage="batch_benchmark", batch_size=size, timed_steps=i + 1, loss=ce)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        result = {
            "batch_size": size,
            "warmup_steps": 7,
            "timed_steps": 64,
            "targets": targets,
            "seconds": elapsed,
            "targets_per_second": targets / elapsed,
            "seconds_per_step": elapsed / 64,
            "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
            "maximum_length_batch_backward_passed": True,
        }
        measurements.append(result)
        event(stage="batch_benchmark_complete", **result)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    speeds = {r["batch_size"]: r["targets_per_second"] for r in measurements}
    selected = 32 if speeds[32] >= 0.95 * speeds[16] else 16
    result = {
        "passed": True,
        "selected_batch_size": selected,
        "measurements": measurements,
        "selection_rule": "prefer paper main batch32 if throughput >=95% of batch16; both must pass worst-length FP32 backward",
        "precision": "fp32",
        "attention": "math",
        "tf32": False,
        "contract": contract(root),
    }
    write(root / "selected_runtime.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=("cpu", "benchmark", "gpu_gate"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if root != ROOT.resolve() or out.parent != root or out.exists():
        raise ValueError("Fresh independent gate directory required")
    if args.job != "cpu":
        cpu = read(root / "cpu_gate.json")
        if (
            not cpu["passed"]
            or cpu["contract"] != contract(root)
            or not os.environ.get("POLAR_SUPERVISOR_PID")
        ):
            raise ValueError("Matching CPU gate and host supervisor required")
        if (root / "GPU_STOP.json").exists() or not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable or stop latched; no fallback")
    out.mkdir()

    def event(**value):
        value["unix_time"] = time.time()
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    event(stage="gate_start", job=args.job)
    began = time.monotonic()
    verify_data(root)
    binding = contract(root)
    with attention_backend("math"):
        if args.job == "benchmark":
            result = benchmark(root, out, event)
        else:
            device = "cpu" if args.job == "cpu" else "cuda"
            sizes = (
                [32]
                if args.job == "cpu"
                else [read(root / "selected_runtime.json")["selected_batch_size"]]
            )
            audits = [backward_resume(root, out, device, size, event, binding) for size in sizes]
            result = {
                "passed": True,
                "contract": binding,
                "interpreter": str(Path(sys.executable).resolve()),
                "device": device,
                "audits": audits,
                "elapsed_seconds": time.monotonic() - began,
            }
            if args.job == "cpu":
                write(root / "cpu_gate.json", result)
            else:
                result["selected_runtime_sha256"] = digest(root / "selected_runtime.json")
                write(root / "gpu_training_gate.json", result)
    write(out / "gate_report.json", result)
    event(stage="gate_passed", job=args.job, elapsed_seconds=time.monotonic() - began)


if __name__ == "__main__":
    main()
