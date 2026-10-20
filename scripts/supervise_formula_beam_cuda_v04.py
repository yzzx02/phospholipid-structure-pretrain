"""Native Windows CUDA inference with a persistent, fail-closed host monitor."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path("outputs/formula_constrained_v04_rtx4070_20261020")
CHECKPOINTS = {
    "fp": Path("outputs/pretraining_v04/runs/fp_smoke_3000/checkpoint/step_003000.pt"),
    "gine": Path("outputs/pretraining_v04/runs/gine_smoke_3000/checkpoint/step_003000.pt"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump_snapshot():
    # These are the GPU timeout locations used by both AMD and NVIDIA Windows drivers.
    result, unreadable = {}, []
    base = Path(os.environ.get("SystemRoot", "C:/Windows")) / "LiveKernelReports"
    for name in ("WATCHDOG", "DISPLAY", "NVIDIA", "AMD_WATCHDOG", "AMD_REPORT_UM"):
        folder = base / name
        try:
            if folder.exists():
                for path in folder.glob("*.dmp"):
                    stat = path.stat()
                    result[str(path)] = [stat.st_mtime_ns, stat.st_size]
        except PermissionError:
            unreadable.append(str(folder))
    return {"files": result, "unreadable_directories": unreadable}


def changed_dumps(baseline, current):
    if baseline["unreadable_directories"] != current["unreadable_directories"]:
        raise RuntimeError("GPU dump monitoring access changed during the experiment")
    return [
        name for name, value in current["files"].items() if baseline["files"].get(name) != value
    ]


def display_faults(since):
    # ISO UTC is parsed explicitly; no dependence on the shell's JSON date conversion.
    # ErrorAction Ignore handles an empty event query. Failure to run PowerShell is fatal.
    script = """
    $since = [DateTimeOffset]::Parse('__SINCE__').UtcDateTime
    Get-WinEvent -ListLog System,Application -ErrorAction Stop | Out-Null
    $events = @(
        Get-WinEvent -FilterHashtable @{LogName='System'; Id=4101;
            StartTime=$since.ToLocalTime()} -ErrorAction Ignore
        Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1001;
            StartTime=$since.ToLocalTime()} -ErrorAction Ignore |
            Where-Object { $_.Message -match 'LiveKernelEvent' -and
                $_.Message -match '(?m)(141|117)' }
    ) | Where-Object { $_.TimeCreated.ToUniversalTime() -ge $since } |
        Select-Object Id, RecordId, ProviderName,
            @{n='utc';e={$_.TimeCreated.ToUniversalTime().ToString('o')}}
    ConvertTo-Json -InputObject @($events) -Compress
    """.replace("__SINCE__", since)
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return json.loads(result.stdout.strip() or "[]")


def telemetry():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,memory.used,"
            "memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("fingerprint", "graph"), required=True)
    parser.add_argument("--mode", choices=("raw", "elements", "elements_grammar"), required=True)
    parser.add_argument("--beam-size", type=int, choices=(5, 20, 50, 128), default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--startup-timeout-seconds", type=int, default=180)
    parser.add_argument("--no-progress-timeout-seconds", type=int, default=20)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This supervisor is only for native Windows")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_name):
        parser.error("Use a simple fresh run name")
    if not 0 <= args.start < 512 or not 1 <= args.count <= 512 - args.start:
        parser.error("Invalid slice of the frozen 512 validation heads")
    if not 30 <= args.timeout_seconds <= 1200:
        parser.error("Per-job timeout must be between 30 and 1200 seconds")
    root = args.experiment_root.resolve()
    protected = [
        Path(p).resolve()
        for p in (
            "outputs/pretraining_v04",
            "outputs/formula_constrained_v04_20261020",
        )
    ]
    if not root.is_relative_to(Path("outputs").resolve()) or any(
        root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected
    ):
        parser.error("Use a separate native CUDA experiment directory under outputs/")
    stop = root / "GPU_STOP.json"
    if stop.exists():
        raise RuntimeError("Native CUDA stop latched; no automatic retry")
    gate = read(root / "cpu_gate.json")
    if (
        not gate["passed"]
        or gate.get("gpu_backend") != "native_cuda"
        or (gate.get("interpreter") != str(Path(sys.executable).resolve()))
    ):
        raise RuntimeError("This interpreter's independent native CUDA CPU gate is required")
    if not gate.get("inference_source_sha256") or any(
        digest(Path(name)) != expected for name, expected in gate["inference_source_sha256"].items()
    ):
        raise RuntimeError("Inference sources changed after the CPU gate")
    run = root / args.run_name
    if run.exists():
        raise FileExistsError("Fresh run name required")
    epoch_path = root / "gpu_monitor_epoch_private.json"
    current_hashes = {key: digest(path) for key, path in CHECKPOINTS.items()}
    audited_hashes = {
        "fp" if audit["encoder"] == "fingerprint" else "gine": audit["inputs"]["checkpoint_sha256"]
        for audit in gate["cpu_checkpoint_audits"]
    }
    if current_hashes != audited_hashes:
        raise RuntimeError("Checkpoints changed after the CPU gate")
    if not epoch_path.exists():
        write_new(
            epoch_path,
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "existing_dumps": dump_snapshot(),
                "checkpoints_sha256": current_hashes,
                "initial_telemetry": telemetry(),
                "cpu_gate_sha256": digest(root / "cpu_gate.json"),
                "historical_AMD_stop_latches_preserved": True,
            },
        )
    epoch = read(epoch_path)
    if epoch["checkpoints_sha256"] != current_hashes or (
        epoch["cpu_gate_sha256"] != digest(root / "cpu_gate.json")
    ):
        raise RuntimeError("Protected experiment inputs changed")

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
        prior_dumps = changed_dumps(epoch["existing_dumps"], dump_snapshot())
        prior_faults = display_faults(epoch["utc"])
        if prior_dumps or prior_faults:
            latch("gpu_fault_since_native_experiment_start")
            raise RuntimeError("A GPU fault occurred since this experiment began")
    except Exception:
        latch("preflight_host_monitor_failure")
        raise
    command = [
        sys.executable,
        "-u",
        "-m",
        "structure_pretrain.polar_only_v04.formula_beam_eval",
        "--encoder",
        args.encoder,
        "--mode",
        args.mode,
        "--beam-size",
        str(args.beam_size),
        "--start",
        str(args.start),
        "--count",
        str(args.count),
        "--device",
        "cuda",
        "--gpu-backend",
        "native_cuda",
        "--experiment-root",
        str(root),
        "--output",
        str(run),
    ]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    # All temporary files and download caches for this workspace live beside the repository.
    workspace = Path.cwd().parent
    for name, folder in (
        ("TEMP", "cache/tmp"),
        ("TMP", "cache/tmp"),
        ("PIP_CACHE_DIR", "cache/pip"),
        ("TORCH_HOME", "cache/torch"),
        ("HF_HOME", "cache/huggingface"),
    ):
        location = workspace / folder
        location.mkdir(parents=True, exist_ok=True)
        env[name] = str(location)
    stdout_path, stderr_path = (
        root / f"{args.run_name}.stdout.log",
        root / f"{args.run_name}.stderr.log",
    )
    reason, new_dumps, faults, samples = None, [], [], []
    started = time.monotonic()
    with (
        stdout_path.open("x", encoding="utf-8") as out,
        stderr_path.open("x", encoding="utf-8") as err,
    ):
        child = subprocess.Popen(
            command, stdout=out, stderr=err, env=env, creationflags=subprocess.CREATE_NO_WINDOW
        )
        print(json.dumps({"stage": "started", "pid": child.pid, "run": args.run_name}), flush=True)
        last_host_poll, last_progress = 0.0, 0
        try:
            while child.poll() is None:
                time.sleep(0.5)
                elapsed = time.monotonic() - started
                if stop.exists():
                    reason = "external_stop_latch"
                    break
                new_dumps = changed_dumps(epoch["existing_dumps"], dump_snapshot())
                if new_dumps:
                    reason = "new_windows_gpu_fault_dump"
                    break
                if elapsed - last_host_poll >= 5:
                    faults = display_faults(epoch["utc"])
                    samples.append({"seconds": elapsed, "gpu": telemetry()})
                    last_host_poll = elapsed
                    if faults:
                        reason = "windows_display_driver_fault"
                        break
                if elapsed > args.timeout_seconds:
                    reason = "total_timeout"
                    break
                events_path = run / "events.jsonl"
                age, decoding = elapsed, False
                if events_path.exists():
                    age = time.time() - events_path.stat().st_mtime
                    content = events_path.read_text()
                    # A writer may be between the JSON payload and its trailing newline.
                    lines = (
                        content.splitlines()
                        if content.endswith("\n")
                        else content.splitlines()[:-1]
                    )
                    events = [json.loads(line) for line in lines]
                    decoding = any(e["stage"] in ("decoding", "target_completed") for e in events)
                    done = [e for e in events if e["stage"] == "target_completed"]
                    if done and done[-1]["targets"] != last_progress:
                        last_progress = done[-1]["targets"]
                        print(json.dumps(done[-1]), flush=True)
                threshold = (
                    args.no_progress_timeout_seconds if decoding else args.startup_timeout_seconds
                )
                if age > threshold:
                    reason = "no_progress_timeout"
                    break
        except BaseException as exc:
            reason = f"host_monitor_exception:{type(exc).__name__}"
        finally:
            if reason:
                latch(reason)
                if child.poll() is None:
                    child.kill()
            child.wait()
    try:
        new_dumps = changed_dumps(epoch["existing_dumps"], dump_snapshot())
        faults = display_faults(epoch["utc"])
        if new_dumps or faults:
            reason = "late_windows_gpu_fault"
        after_hashes = {key: digest(path) for key, path in CHECKPOINTS.items()}
        if after_hashes != epoch["checkpoints_sha256"]:
            reason = "checkpoint_integrity_failure"
    except Exception as exc:
        after_hashes = None
        reason = f"final_monitor_exception:{type(exc).__name__}"
    if child.returncode and not reason:
        reason = "child_failed"
    if reason:
        latch(reason)
    run.mkdir(exist_ok=True)
    stdout_path.rename(run / "stdout.log")
    stderr_path.rename(run / "stderr.log")
    report = {
        "child_exit_code": child.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stop_reason": reason,
        "new_windows_gpu_dump_count": len(new_dumps),
        "display_reset_count": len(faults),
        "gpu_fault_events": faults,
        "new_dump_paths_private": new_dumps,
        "gpu_telemetry": samples,
        "encoder": args.encoder,
        "mode": args.mode,
        "beam_size": args.beam_size,
        "start": args.start,
        "count": args.count,
        "training_started": False,
        "gpu_backend": "native_cuda",
        "attention_backend": "math",
        "precision": "fp32",
        "per_job_timeout_seconds": args.timeout_seconds,
        "startup_timeout_seconds": args.startup_timeout_seconds,
        "no_progress_timeout_seconds": args.no_progress_timeout_seconds,
        "experiment_monitor_start_utc": epoch["utc"],
        "checkpoints_after_sha256": after_hashes,
        "checkpoints_unchanged": after_hashes == epoch["checkpoints_sha256"],
        "dump_monitor_unreadable_directories": epoch["existing_dumps"]["unreadable_directories"],
        "event_monitor": "System 4101 + Application WER LiveKernelEvent 141/117",
    }
    write_new(run / "host_monitor.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "child_exit_code",
                    "elapsed_seconds",
                    "stop_reason",
                    "new_windows_gpu_dump_count",
                    "display_reset_count",
                    "checkpoints_unchanged",
                )
            }
        ),
        flush=True,
    )
    if reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
