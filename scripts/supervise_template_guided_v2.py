"""Independent native Windows GPU monitor; fail closed, never automatically retry."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from supervise_formula_beam_cuda_v04 import (
    CHECKPOINTS,
    changed_dumps,
    display_faults,
    dump_snapshot,
    telemetry,
    write_new,
)

from structure_pretrain.formal_v2.common import ROOT, contract, digest, read, write


def old_file_inventory():
    paths = []
    for old in ("template_guided_fp5000_20261022", "template_guided_fp5000_20261022_r1"):
        paths.extend((Path("outputs") / old).glob("**/*.pt"))
    return {
        str(p.resolve()): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in paths
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=("cuda_gate", "train", "evaluate"), required=True)
    parser.add_argument("--experiment", choices=("E1", "E2", "E3"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=172800)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--cohort",
        choices=("validation_pilot", "validation_full", "test"),
        default="validation_pilot",
    )
    args = parser.parse_args()
    root = ROOT.resolve()
    if sys.platform != "win32" or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("Native Windows and dedicated V2 run name required")
    if not 60 <= args.timeout_seconds <= 172800:
        raise ValueError("Bounded timeout required")
    run = root / args.run_name
    stop = root / "GPU_STOP.json"
    if stop.exists() or run.exists():
        raise ValueError("Fresh run and no latched GPU fault required; no automatic retry")
    binding = contract(root)
    cpu = read(root / "cpu_gate.json")
    if (
        not cpu["passed"]
        or cpu["contract"] != binding
        or cpu["interpreter"] != str(Path(sys.executable).resolve())
    ):
        raise ValueError("Matching CPU gate required")
    if args.job != "cuda_gate":
        cuda = read(root / "cuda_gate.json")
        if not cuda["passed"] or cuda["contract"] != binding:
            raise ValueError("Matching real CUDA resume gate required")
    protected_paths = list(CHECKPOINTS.values()) + sorted(
        Path("outputs/polar_fp_base_50k_20261020").glob("*/checkpoint/*.pt")
    )
    best = Path("outputs/template_guided_fp5000_20261022_r1/train_main_k3_seed42/best.json")
    protected_paths.append(Path(read(best)["path"]))
    if args.checkpoint:
        protected_paths.append(args.checkpoint.resolve())
    if args.resume:
        protected_paths.append(args.resume.resolve())
    protected = {str(p.resolve()): digest(p) for p in protected_paths}
    old_inventory = old_file_inventory()
    epoch_path = root / "gpu_monitor_epoch_private.json"
    if not epoch_path.exists():
        write_new(
            epoch_path,
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "existing_dumps": dump_snapshot(),
                "initial_telemetry": telemetry(),
                "protected_checkpoint_sha256": protected,
                "all_exploratory_weights_inventory": old_inventory,
                "cpu_gate_sha256": digest(root / "cpu_gate.json"),
                "contract": binding,
            },
        )
    epoch = read(epoch_path)
    if epoch["contract"] != binding or epoch["cpu_gate_sha256"] != digest(root / "cpu_gate.json"):
        raise ValueError("Frozen code/config/gate changed after monitor start")
    if epoch["all_exploratory_weights_inventory"] != old_inventory:
        raise ValueError("Exploratory weights changed since monitor epoch began")

    def latch(reason):
        if not stop.exists():
            write_new(
                stop,
                {
                    "reason": reason,
                    "run": args.run_name,
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )

    try:
        for path, expected in epoch["protected_checkpoint_sha256"].items():
            if digest(path) != expected:
                raise ValueError("Historical checkpoint changed")
        if changed_dumps(epoch["existing_dumps"], dump_snapshot()) or display_faults(epoch["utc"]):
            raise ValueError("GPU fault since monitor epoch start")
    except Exception:
        latch("preflight_host_monitor_failure")
        raise
    if args.job == "cuda_gate":
        command = [sys.executable, "-u", "scripts/gate_template_guided_v2.py", "--device", "cuda"]
    elif args.job == "train":
        if not args.experiment:
            raise ValueError("Explicit experiment required")
        command = [
            sys.executable,
            "-u",
            "scripts/train_template_guided_v2.py",
            "--experiment",
            args.experiment,
        ]
        if args.resume:
            command += ["--resume", str(args.resume.resolve())]
    else:
        if not args.checkpoint:
            raise ValueError("Explicit evaluation checkpoint required")
        command = [
            sys.executable,
            "-u",
            "scripts/evaluate_template_guided_v2.py",
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--cohort",
            args.cohort,
        ]
    command += ["--output", str(run)]
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONUNBUFFERED="1",
        POLAR_SUPERVISOR_PID=str(os.getpid()),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    if Path(env.get("TEMP", "")).drive.upper() != "G:":
        raise ValueError("Use enter_cuda_v04.ps1 for G-drive caches")
    stdout, stderr = root / f"{args.run_name}.stdout.log", root / f"{args.run_name}.stderr.log"
    status = root / f"{args.run_name}.supervisor.json"
    began = time.monotonic()
    last_poll = 0.0
    reason, faults, new_dumps, samples = None, [], [], []
    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        child = subprocess.Popen(
            command, stdout=out, stderr=err, env=env, creationflags=subprocess.CREATE_NO_WINDOW
        )
        state = {
            "status": "running",
            "job": args.job,
            "experiment": args.experiment,
            "supervisor_pid": os.getpid(),
            "child_pid": child.pid,
            "run": str(run),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": args.timeout_seconds,
            "auto_retry": False,
        }
        write(status, state)
        print(json.dumps(state), flush=True)
        try:
            while child.poll() is None:
                time.sleep(0.5)
                elapsed = time.monotonic() - began
                if stop.exists():
                    reason = "external_stop_latch"
                    break
                if elapsed - last_poll >= 10:
                    faults = display_faults(epoch["utc"])
                    new_dumps = changed_dumps(epoch["existing_dumps"], dump_snapshot())
                    samples.append({"seconds": elapsed, "gpu": telemetry()})
                    write(
                        status,
                        {**state, "elapsed_seconds": elapsed, "latest_telemetry": samples[-1]},
                    )
                    last_poll = elapsed
                    if faults or new_dumps:
                        reason = "windows_GPU_fault_or_dump"
                        break
                if elapsed > args.timeout_seconds:
                    reason = "total_timeout"
                    break
                events = run / "events.jsonl"
                age = time.time() - events.stat().st_mtime if events.exists() else elapsed
                if age > (240 if elapsed < 300 else 120):
                    reason = "no_progress_timeout"
                    break
        except BaseException as exc:
            reason = "host_monitor_exception:" + type(exc).__name__
        finally:
            if reason:
                latch(reason)
                if child.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        capture_output=True,
                        timeout=15,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    if child.poll() is None:
                        child.kill()
            child.wait()
    try:
        faults = display_faults(epoch["utc"])
        new_dumps = changed_dumps(epoch["existing_dumps"], dump_snapshot())
        unchanged = all(digest(p) == value for p, value in protected.items())
        inventory_unchanged = old_file_inventory() == old_inventory
        if faults or new_dumps:
            reason = "late_GPU_fault"
        if not unchanged or not inventory_unchanged:
            reason = "protected_weight_changed"
    except Exception as exc:
        unchanged, inventory_unchanged = False, False
        reason = "final_monitor_exception:" + type(exc).__name__
    if child.returncode and not reason:
        reason = "child_failed"
    if reason:
        latch(reason)
    run.mkdir(exist_ok=True)
    stdout.rename(run / "stdout.log")
    stderr.rename(run / "stderr.log")
    report = {
        "child_exit_code": child.returncode,
        "stop_reason": reason,
        "elapsed_seconds": time.monotonic() - began,
        "display_reset_count": len(faults),
        "new_GPU_dump_count": len(new_dumps),
        "gpu_fault_events": faults,
        "telemetry": samples,
        "protected_checkpoint_sha256": protected,
        "protected_checkpoints_unchanged": unchanged,
        "all_exploratory_weights_inventory_unchanged": inventory_unchanged,
        "protected_exploratory_checkpoint_count": len(old_inventory),
        "unreadable_dump_directories": epoch["existing_dumps"]["unreadable_directories"],
        "event_monitor": "System4101; WER LiveKernelEvent141/117",
        "auto_retry": False,
    }
    write_new(run / "host_monitor.json", report)
    write(status, {**state, "status": "failed" if reason else "completed", "stop_reason": reason})
    if reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
