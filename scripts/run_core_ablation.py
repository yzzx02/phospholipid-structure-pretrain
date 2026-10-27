"""Serial supervised full-test constraints and a separately trained Formula-only control."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from structure_pretrain.core_ablation.common import (
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
        raise ValueError("No duplicate queue or automatic GPU retry")
    binding = contract(root)
    for name in ("cpu_gate.json", "cuda_gate.json"):
        gate = read(root / name)
        assert gate["passed"] and gate["contract"] == binding
    with (root / "queue_started.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "contract": binding,
                "configuration": c,
                "queue_pid": os.getpid(),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "authorization": (
                    "2026-10-27 user requests full9499, Formula-only "
                    "and prioritizes actual online formula constraints"
                ),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )

    def report():
        result = subprocess.run([sys.executable, "scripts/report_core_ablation.py"])
        if result.returncode:
            write(
                root / "REPORT_PENDING.json",
                {"reason": "CPU aggregate reporter failed", "GPU_queue_continues": True},
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
                "scripts/supervise_core_ablation.py",
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

    # Real online formula + grammar comes first, as explicitly prioritized.
    order = ("elements_grammar", "raw", "elements")
    for mode in order:
        job(
            "evaluate",
            "template_full_" + mode,
            [
                "--checkpoint",
                c["template_checkpoint"],
                "--model-kind",
                "template",
                "--decode-mode",
                mode,
            ],
        )
    job("train", "formula_only_train_seed42", ["--experiment", "formula_only"])
    best = read(root / "formula_only_train_seed42/best.json")
    history = read(root / "formula_only_train_seed42/validation_history.json")
    assert best["best_validation_loss"] == min(r["validation_token_CE"] for r in history)
    assert digest(best["path"]) == best["sha256"]
    for mode in order:
        job(
            "evaluate",
            "formula_only_full_" + mode,
            ["--checkpoint", best["path"], "--model-kind", "formula_only", "--decode-mode", mode],
        )
    write(
        root / "state.json",
        {
            "status": "completed",
            "stage": "full_test_and_formula_control_completed",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "test_used_for_selection": False,
        },
    )
    report()


if __name__ == "__main__":
    main()
