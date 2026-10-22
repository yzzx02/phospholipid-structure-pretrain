"""Independent fail-closed Windows host supervisor for the authorized fixed-5k run."""

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

from structure_pretrain.reported_headgroups.formal_template_data import (
    ROOT,
    contract,
    digest,
    read,
    write,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=("benchmark", "gpu_gate", "train"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if (
        sys.platform != "win32"
        or root != ROOT.resolve()
        or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name)
    ):
        raise ValueError("Native Windows and dedicated fresh root/run required")
    if not 60 <= args.timeout_seconds <= 172800:
        raise ValueError("Bounded timeout must be 60 seconds to 48 hours")
    run = root / args.run_name
    if run.exists():
        raise FileExistsError(run)
    stop = root / "GPU_STOP.json"
    if stop.exists():
        raise RuntimeError("GPU stop latched; never retry automatically")
    recovery = root / "GPU_RESTART_AUTHORIZATION.json"
    if recovery.exists() and not read(recovery)["authorized"]:
        raise RuntimeError(
            "Prior failed GPU attempt retained; explicit user retry authorization required"
        )
    binding = contract(root)
    gate = read(root / "cpu_gate.json")
    if (
        not gate["passed"]
        or gate["contract"] != binding
        or gate["interpreter"] != str(Path(sys.executable).resolve())
    ):
        raise ValueError("Matching CPU gate required")
    if args.job == "train":
        gpu = read(root / "gpu_training_gate.json")
        if (
            not gpu["passed"]
            or gpu["contract"] != binding
            or gpu["selected_runtime_sha256"] != digest(root / "selected_runtime.json")
        ):
            raise ValueError("Real CUDA backward/restart and batch benchmark required")
    protected_paths = list(CHECKPOINTS.values())
    # Preserve every historical FP50k checkpoint, including probes, without writing into that tree.
    protected_paths += sorted(Path("outputs/polar_fp_base_50k_20261020").glob("*/checkpoint/*.pt"))
    protected = {str(p.resolve()): digest(p) for p in protected_paths}
    if args.resume:
        protected[str(args.resume.resolve())] = digest(args.resume)
    epoch_path = root / "gpu_monitor_epoch_private.json"
    if not epoch_path.exists():
        write_new(
            epoch_path,
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "existing_dumps": dump_snapshot(),
                "initial_telemetry": telemetry(),
                "protected_checkpoint_sha256": protected,
                "cpu_gate_sha256": digest(root / "cpu_gate.json"),
                "contract": binding,
            },
        )
    epoch = read(epoch_path)
    if epoch["cpu_gate_sha256"] != digest(root / "cpu_gate.json") or epoch["contract"] != binding:
        raise ValueError("CPU gate/source changed after monitor epoch began")
    for path, expected in epoch["protected_checkpoint_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Historical checkpoint changed")

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
        if changed_dumps(epoch["existing_dumps"], dump_snapshot()) or display_faults(epoch["utc"]):
            raise RuntimeError("GPU fault since epoch start")
    except Exception:
        latch("preflight_host_monitor_failure")
        raise
    if args.job == "train":
        command = [sys.executable, "-u", "train_template_guided_fp5000.py"]
        if args.resume:
            command += ["--resume", str(args.resume.resolve())]
    else:
        command = [
            sys.executable,
            "-u",
            "scripts/gate_template_guided_fp5000.py",
            "--job",
            args.job,
        ]
    command += ["--root", str(root), "--output", str(run)]
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONUNBUFFERED="1",
        POLAR_SUPERVISOR_PID=str(os.getpid()),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    if Path(env.get("TEMP", "")).drive.upper() != "G:":
        raise ValueError("Dot-source enter_cuda_v04.ps1 to keep caches/temp on G")
    stdout, stderr = root / f"{args.run_name}.stdout.log", root / f"{args.run_name}.stderr.log"
    status_path = root / f"{args.run_name}.supervisor.json"
    began, last_poll, last_report = time.monotonic(), 0.0, 0.0
    reason, faults, new_dumps, samples = None, [], [], []
    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        child = subprocess.Popen(
            command, stdout=out, stderr=err, env=env, creationflags=subprocess.CREATE_NO_WINDOW
        )
        state = {
            "status": "running",
            "job": args.job,
            "supervisor_pid": os.getpid(),
            "child_pid": child.pid,
            "run": str(run),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": args.timeout_seconds,
            "monitor_epoch_utc": epoch["utc"],
            "auto_retry": False,
        }
        write(status_path, state)
        print(json.dumps(state), flush=True)
        try:
            while child.poll() is None:
                time.sleep(0.5)
                elapsed = time.monotonic() - began
                if stop.exists():
                    reason = "external_stop_latch"
                    break
                new_dumps = changed_dumps(epoch["existing_dumps"], dump_snapshot())
                if new_dumps:
                    reason = "new_gpu_dump"
                    break
                if elapsed - last_poll >= 10:
                    faults = display_faults(epoch["utc"])
                    samples.append({"seconds": elapsed, "gpu": telemetry()})
                    last_poll = elapsed
                    write(
                        status_path,
                        {**state, "last_monitor_seconds": elapsed, "latest_telemetry": samples[-1]},
                    )
                    if faults:
                        reason = "windows_display_driver_fault"
                        break
                if elapsed > args.timeout_seconds:
                    reason = "total_timeout"
                    break
                events = run / "events.jsonl"
                age = time.time() - events.stat().st_mtime if events.exists() else elapsed
                if age > (180 if elapsed < 240 else 90):
                    reason = "no_progress_timeout"
                    break
                if elapsed - last_report >= 30 and events.exists():
                    with events.open(encoding="utf-8") as stream:
                        lines = stream.readlines()
                    if lines:
                        print(lines[-1].strip(), flush=True)
                    last_report = elapsed
        except BaseException as exc:
            reason = f"host_monitor_exception:{type(exc).__name__}"
        finally:
            if reason:
                latch(reason)
                if child.poll() is None:
                    # Windows venv's launcher can own a real Python child; stop the entire GPU tree.
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
        after = {p: digest(p) for p in protected}
        if faults or new_dumps:
            reason = "late_gpu_fault"
        if after != protected:
            reason = "protected_checkpoint_changed"
    except Exception as exc:
        after = None
        reason = f"final_host_monitor_exception:{type(exc).__name__}"
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
        "job": args.job,
        "elapsed_seconds": time.monotonic() - began,
        "display_reset_count": len(faults),
        "new_gpu_dump_count": len(new_dumps),
        "gpu_fault_events": faults,
        "telemetry": samples,
        "protected_checkpoints_sha256": protected,
        "protected_checkpoints_unchanged": after == protected,
        "monitor_epoch_utc": epoch["utc"],
        "timeout_seconds": args.timeout_seconds,
        "auto_retry": False,
        "unreadable_dump_directories": epoch["existing_dumps"]["unreadable_directories"],
        "event_monitor": "System4101; Application WER LiveKernelEvent141/117",
    }
    write_new(run / "host_monitor.json", report)
    write(
        status_path,
        {
            **state,
            "status": "failed" if reason else "completed",
            "stop_reason": reason,
            "elapsed_seconds": report["elapsed_seconds"],
        },
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "child_exit_code",
                    "stop_reason",
                    "elapsed_seconds",
                    "protected_checkpoints_unchanged",
                )
            }
        ),
        flush=True,
    )
    if reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
