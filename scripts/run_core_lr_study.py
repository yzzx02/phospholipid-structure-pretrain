"""Serial, supervised schedule comparison; no retries or test evaluation."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from structure_pretrain.core_lr_study.common import (
    ROOT,
    configuration,
    contract,
    digest,
    read,
    write,
)


def main():
    root, c = ROOT.resolve(), configuration()
    if (root / "queue_started.json").exists() or (root / "GPU_STOP.json").exists():
        raise ValueError("No duplicate queue or automatic retry")
    binding = contract(root)
    for name in ("cpu_gate.json", "cuda_gate.json"):
        assert read(root / name)["passed"] and read(root / name)["contract"] == binding
    with (root / "queue_started.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "contract": binding,
                "configuration": c,
                "queue_pid": os.getpid(),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "authorization": "2026-10-28 user LR comparison; common real1-50 plots; early-stop "
                "selection",
                "policy": "minimum50, patience5 without reset at50, maximum80; global "
                "lowest full validation CE",
            },
            stream,
            indent=2,
        )

    def report():
        result = subprocess.run([sys.executable, "scripts/report_core_lr_study.py"])
        if result.returncode:
            write(
                root / "REPORT_PENDING.json",
                {"reason": "CPU reporter failed", "GPU_queue_continues": True},
            )

    def job(kind, name, extra):
        write(
            root / "state.json",
            {
                "status": "running",
                "stage": name,
                "queue_pid": os.getpid(),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "scripts/supervise_core_lr_study.py",
                "--job",
                kind,
                "--run-name",
                name,
                "--timeout-seconds",
                str(c["timeout_seconds"]),
                *extra,
            ]
        )
        if result.returncode:
            write(
                root / "state.json",
                {
                    "status": "failed",
                    "stage": name,
                    "auto_retry": False,
                    "exit_code": result.returncode,
                },
            )
            report()
            raise SystemExit(result.returncode)
        report()

    # Keep original E3 intact; obtain genuinely measured epochs49+ in a new run.
    for exp in ("E3", "E1", "E2", "E4"):
        job("train", f"{exp}_seed42", ["--experiment", exp])
    for exp in ("E1", "E2", "E3", "E4"):
        best = read(root / f"{exp}_seed42/best.json")
        history = read(root / f"{exp}_seed42/validation_history.json")
        assert len(history) >= 50
        assert best["best_validation_loss"] == min(r["validation_token_CE"] for r in history)
        assert digest(best["path"]) == best["sha256"]
        job(
            "evaluate",
            f"{exp}_validation_beam32",
            ["--experiment", exp, "--checkpoint", best["path"]],
        )
    write(
        root / "state.json",
        {
            "status": "completed",
            "stage": "four_schedules_and_validation_Beam32_complete",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "test_used_for_selection": False,
        },
    )
    report()
    # Publication is CPU-only and cannot restart a failed GPU job.
    subprocess.run([sys.executable, "scripts/publish_lr_schedule_results.py"], check=True)


if __name__ == "__main__":
    main()
