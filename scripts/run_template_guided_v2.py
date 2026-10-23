"""One bounded serial V2 queue; any failed child stops the queue without retry."""

import subprocess
import sys

from structure_pretrain.formal_v2.common import ROOT, contract, digest, emit, read, write


def main():
    root = ROOT.resolve()
    if (root / "queue_started.json").exists() or (root / "GPU_STOP.json").exists():
        raise ValueError("Queue already started or GPU stop latched; never retry automatically")
    binding = contract(root)
    gate = read(root / "cpu_gate.json")
    if not gate["passed"] or gate["contract"] != binding:
        raise ValueError("Matching CPU gate required")
    write(
        root / "queue_started.json",
        {
            "contract": binding,
            "experiments": ["E1", "E2", "E3"],
            "fresh_initialization_seed": 42,
            "automatic_retry": False,
            "Formula_only": (
                "conditional later stage after template training and generation "
                "assessment; not started by this queue"
            ),
        },
    )

    def job(name, *arguments):
        if (root / "GPU_STOP.json").exists():
            raise ValueError("GPU stop latched")
        write(root / "state.json", {"stage": name, "status": "running", "automatic_retry": False})
        emit(stage="queue_job_started", job=name)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "scripts/supervise_template_guided_v2.py",
                "--run-name",
                name,
                *arguments,
            ],
            check=True,
        )
        monitor = read(root / name / "host_monitor.json")
        if monitor["stop_reason"] or monitor["child_exit_code"]:
            raise ValueError("Supervised job failed; stop queue")
        emit(stage="queue_job_completed", job=name)

    job("gate_cuda", "--job", "cuda_gate", "--timeout-seconds", "3600")
    for experiment in ("E1", "E2", "E3"):
        job(experiment + "_seed42", "--job", "train", "--experiment", experiment)
    reports = {
        experiment: read(root / (experiment + "_seed42/training_report.json"))
        for experiment in ("E1", "E2", "E3")
    }
    selected = min(reports, key=lambda exp: reports[exp]["best_validation_token_CE"])
    pointer = read(root / (selected + "_seed42/best.json"))
    if digest(pointer["path"]) != pointer["sha256"]:
        raise ValueError("Best selected checkpoint hash mismatch")
    write(
        root / "LR_selection.json",
        {
            "selected_experiment": selected,
            "reports": reports,
            "criterion": "minimum complete validation Token CE",
            "test_used": False,
            "best": pointer,
        },
    )
    for cohort in ("validation_pilot", "validation_full", "test"):
        job(
            "eval_" + cohort,
            "--job",
            "evaluate",
            "--checkpoint",
            pointer["path"],
            "--cohort",
            cohort,
        )
        report = read(root / ("eval_" + cohort) / "evaluation_report.json")
        if (
            report["targets"]
            != {"validation_pilot": 300, "validation_full": 9527, "test": 9548}[cohort]
        ):
            raise ValueError("Evaluation target denominator changed")
    write(
        root / "state.json",
        {
            "stage": "template_E1_E2_E3_and_generation_complete",
            "status": "completed_template_phase",
            "selected_experiment": selected,
            "Formula_only": (
                "pending assessment of template stability and independent fair control design"
            ),
            "extra_seeds": "pending selected-optimizer stability assessment",
            "automatic_retry": False,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write(
            ROOT / "queue_failure.json",
            {"type": type(exc).__name__, "message": str(exc), "automatic_retry": False},
        )
        raise
