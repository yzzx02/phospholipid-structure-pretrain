"""One supervised new-library training run, then bounded predeclared generation cohorts."""

import os
import subprocess
import sys
from datetime import datetime, timezone

from structure_pretrain.core_templates.common import (
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
        raise ValueError("Do not duplicate or automatically restart a stopped new experiment")
    binding = contract(root)
    for name in ("cpu_gate.json", "cuda_gate.json"):
        proof = read(root / name)
        if not proof["passed"] or proof["contract"] != binding:
            raise ValueError("Matching full-state backward/restart gates required")
    state = {
        "queue_pid": os.getpid(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "contract": binding,
        "configuration": c,
        "data_manifest_sha256": digest(root / "data/manifest.json"),
        "authorization": (
            "2026-10-26 user directs new-library formal training with about8000 templates, "
            "split targets and predict bank entries too"
        ),
    }
    with (root / "queue_started.json").open("x", encoding="utf-8") as stream:
        import json

        json.dump(state, stream, ensure_ascii=False, indent=2)

    def run(job, name, extra):
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
                "scripts/supervise_core_templates8000.py",
                "--job",
                job,
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
                    "exit_code": result.returncode,
                    "automatic_retry": False,
                },
            )
            raise SystemExit(result.returncode)

    run("train", "train_main_k3_seed42", ["--experiment", "core8000"])
    best = read(root / "train_main_k3_seed42/best.json")
    history = read(root / "train_main_k3_seed42/validation_history.json")
    if best["best_validation_loss"] != min(r["validation_token_CE"] for r in history):
        raise ValueError("Best pointer differs from minimum complete-validation CE")
    for cohort in ("validation512", "bank_targets512", "test512"):
        run("evaluate", "eval_" + cohort, ["--checkpoint", best["path"], "--cohort", cohort])
    write(
        root / "state.json",
        {
            "status": "completed",
            "stage": "predeclared_evaluations_completed",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "selected_checkpoint_sha256": best["sha256"],
            "test_used_for_model_selection": False,
        },
    )


if __name__ == "__main__":
    main()
