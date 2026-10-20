"""Fail-closed native training/evaluation monitor; persistent epoch, no retry."""

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
    digest,
    display_faults,
    dump_snapshot,
    read,
    telemetry,
    write_new,
)

from structure_pretrain.polar_only_v04.fp50k_data import ROOT, get_contract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, default=ROOT)
    p.add_argument("--run-name", required=True)
    p.add_argument("--job", choices=("train", "eval"), required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--mode", choices=("raw", "elements_grammar"), default="elements_grammar")
    p.add_argument("--beam-size", type=int, choices=(1, 5, 20, 50), default=20)
    p.add_argument("--count", type=int, default=512)
    p.add_argument("--timeout-seconds", type=int, default=3600)
    args = p.parse_args()
    if sys.platform != "win32" or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        p.error("Native Windows and a fresh simple run name required")
    if not 30 <= args.timeout_seconds <= 7200:
        p.error("Bounded timeout must be 30..7200 seconds")
    root = args.experiment_root.resolve()
    if not root.is_relative_to(Path("outputs").resolve()) or root.name != ROOT.name:
        p.error("Use the dedicated FP50k root")
    run = root / args.run_name
    if run.exists():
        raise FileExistsError(run)
    stop = root / "GPU_STOP.json"
    if stop.exists():
        raise RuntimeError("FP50k stop latched; no automatic retry")
    _, _, contract = get_contract(root)
    gate = read(root / "cpu_gate.json")
    if (
        not gate["passed"]
        or gate["contract"] != contract
        or gate["interpreter"] != str(Path(sys.executable).resolve())
    ):
        raise RuntimeError("Matching independent CPU training gate required")
    if args.job == "train" and not args.probe:
        if args.steps > 1000:
            decision = read(root / "stage_A_decision.json")
            if not decision["continue"] or decision["contract"] != contract:
                raise ValueError("Stage A engineering decision required")
        if args.steps > 3000:
            decision = read(root / "stage_B_decision.json")
            if not decision["continue"] or decision["contract"] != contract:
                raise ValueError("Stage B scientific decision required")
    protected = {str(path.resolve()): digest(path) for path in CHECKPOINTS.values()}
    for path in (args.resume, args.checkpoint):
        if path:
            protected[str(path.resolve())] = digest(path)
    epoch_path = root / "gpu_monitor_epoch_private.json"
    if not epoch_path.exists():
        write_new(
            epoch_path,
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "existing_dumps": dump_snapshot(),
                "initial_telemetry": telemetry(),
                "historical_checkpoints": {str(v): digest(v) for v in CHECKPOINTS.values()},
                "cpu_gate_sha256": digest(root / "cpu_gate.json"),
            },
        )
    epoch = read(epoch_path)
    if epoch["cpu_gate_sha256"] != digest(root / "cpu_gate.json"):
        raise RuntimeError("CPU gate changed after epoch start")
    if epoch["historical_checkpoints"] != {str(v): digest(v) for v in CHECKPOINTS.values()}:
        raise RuntimeError("Historical checkpoint changed since experiment start")

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
            raise RuntimeError("GPU fault since experiment epoch")
    except Exception:
        latch("preflight_host_monitor_failure")
        raise
    if args.job == "train":
        command = [
            sys.executable,
            "-u",
            "train_polar_fp50k.py",
            "--device",
            "cuda",
            "--steps",
            str(args.steps),
        ]
        if args.probe:
            command += ["--probe"]
        if args.resume:
            command += ["--resume", str(args.resume.resolve())]
    else:
        if not args.checkpoint:
            p.error("Evaluation needs a checkpoint")
        command = [
            sys.executable,
            "-u",
            "-m",
            "structure_pretrain.polar_only_v04.fp50k_eval",
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--mode",
            args.mode,
            "--beam-size",
            str(args.beam_size),
            "--count",
            str(args.count),
        ]
    command += ["--experiment-root", str(root), "--output", str(run)]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    for name, folder in (
        ("TEMP", "tmp"),
        ("TMP", "tmp"),
        ("PIP_CACHE_DIR", "pip"),
        ("TORCH_HOME", "torch"),
        ("HF_HOME", "huggingface"),
    ):
        env[name] = str(Path.cwd().parent / "cache" / folder)
    out_path, err_path = root / f"{args.run_name}.stdout.log", root / f"{args.run_name}.stderr.log"
    started = time.monotonic()
    reason, faults, new_dumps, samples, last_poll, last_report = None, [], [], [], 0.0, 0.0
    with out_path.open("x", encoding="utf-8") as out, err_path.open("x", encoding="utf-8") as err:
        child = subprocess.Popen(
            command, stdout=out, stderr=err, env=env, creationflags=subprocess.CREATE_NO_WINDOW
        )
        print(json.dumps({"stage": "started", "pid": child.pid, "run": args.run_name}), flush=True)
        try:
            while child.poll() is None:
                time.sleep(0.5)
                elapsed = time.monotonic() - started
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
                    if faults:
                        reason = "windows_display_driver_fault"
                        break
                if elapsed > args.timeout_seconds:
                    reason = "total_timeout"
                    break
                events = run / "events.jsonl"
                age = time.time() - events.stat().st_mtime if events.exists() else elapsed
                if age > (180 if elapsed < 240 else 60):
                    reason = "no_progress_timeout"
                    break
                if elapsed - last_report >= 30 and events.exists():
                    lines = events.read_text().splitlines()
                    if lines:
                        try:
                            print(lines[-1], flush=True)
                        except ValueError:
                            pass
                    last_report = elapsed
        except BaseException as exc:
            reason = f"host_monitor_exception:{type(exc).__name__}"
        finally:
            if reason:
                latch(reason)
                if child.poll() is None:
                    child.kill()
            child.wait()
    try:
        faults = display_faults(epoch["utc"])
        new_dumps = changed_dumps(epoch["existing_dumps"], dump_snapshot())
        after = {path: digest(Path(path)) for path in protected}
        if faults or new_dumps:
            reason = "late_gpu_fault"
        if after != protected:
            reason = "protected_checkpoint_changed"
    except Exception as exc:
        after = None
        reason = f"final_monitor_exception:{type(exc).__name__}"
    if child.returncode and not reason:
        reason = "child_failed"
    if reason:
        latch(reason)
    run.mkdir(exist_ok=True)
    out_path.rename(run / "stdout.log")
    err_path.rename(run / "stderr.log")
    report = {
        "child_exit_code": child.returncode,
        "stop_reason": reason,
        "job": args.job,
        "elapsed_seconds": time.monotonic() - started,
        "display_reset_count": len(faults),
        "new_gpu_dump_count": len(new_dumps),
        "gpu_fault_events": faults,
        "telemetry": samples,
        "protected_checkpoints_sha256": protected,
        "protected_checkpoints_unchanged": after == protected,
        "monitor_epoch_utc": epoch["utc"],
        "unreadable_dump_directories": epoch["existing_dumps"]["unreadable_directories"],
        "event_monitor": "System 4101; Application WER LiveKernelEvent141/117",
        "timeout_seconds": args.timeout_seconds,
        "auto_retry": False,
    }
    write_new(run / "host_monitor.json", report)
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
