"""User-authorized continuation of the interrupted V2 queue; no automatic retries."""

import subprocess
import sys

from structure_pretrain.formal_v2.common import ROOT, contract, digest, emit, read, write


def main():
    root = ROOT.resolve()
    authorization = read(root / "RESUME_AUTHORIZATION_20261026.json")
    pointer = authorization["resume_checkpoint"]
    binding = contract(root)
    if (
        not authorization["original_source_contract_unchanged"]
        or digest(pointer["path"]) != pointer["sha256"]
        or binding != read(root / "queue_started.json")["contract"]
        or (root / "GPU_STOP.json").exists()
        or (root / "resume_queue_started_20261026.json").exists()
    ):
        raise ValueError("Fresh authorized continuation, unchanged checkpoint and no stop required")
    continuation = root / "resume_queue_started_20261026.json"
    with continuation.open("x", encoding="utf-8") as stream:
        import json

        json.dump(
            {
                "authorization": authorization,
                "prior_state": read(root / "state.json"),
                "contract": binding,
                "automatic_retry": False,
            },
            stream,
            indent=2,
        )

    def job(name, *arguments):
        if (root / "GPU_STOP.json").exists():
            raise ValueError("GPU fault latched; preserve and stop")
        write(
            root / "state.json",
            {
                "stage": name,
                "status": "running",
                "automatic_retry": False,
                "user_authorized_continuation": "2026-10-26",
            },
        )
        emit(stage="authorized_resume_queue_job_started", job=name)
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
            raise ValueError("Continuation supervised job failed; no automatic retry")
        emit(stage="authorized_resume_queue_job_completed", job=name)

    e2 = "E2_seed42_resume_20261026"
    e3 = "E3_seed42_resume_queue_20261026"
    job(e2, "--job", "train", "--experiment", "E2", "--resume", pointer["path"])
    job(e3, "--job", "train", "--experiment", "E3")
    runs = {"E1": "E1_seed42", "E2": e2, "E3": e3}
    reports = {key: read(root / run / "training_report.json") for key, run in runs.items()}
    selected = min(reports, key=lambda key: reports[key]["best_validation_token_CE"])
    best = read(root / runs[selected] / "best.json")
    if digest(best["path"]) != best["sha256"]:
        raise ValueError("Selected best checkpoint changed")
    write(
        root / "LR_selection_resumed_20261026.json",
        {
            "selected_experiment": selected,
            "reports": reports,
            "runs": runs,
            "best": best,
            "criterion": "minimum complete validation Token CE",
            "test_used": False,
        },
    )
    for cohort in ("validation_pilot", "validation_full", "test"):
        name = "eval_" + cohort + "_resume_20261026"
        job(name, "--job", "evaluate", "--checkpoint", best["path"], "--cohort", cohort)
        report = read(root / name / "evaluation_report.json")
        expected = {"validation_pilot": 300, "validation_full": 9527, "test": 9548}[cohort]
        if report["targets"] != expected:
            raise ValueError("Frozen evaluation denominator changed")
    write(
        root / "state.json",
        {
            "stage": "resumed_template_E1_E2_E3_and_generation_complete",
            "status": "completed_template_phase",
            "selected_experiment": selected,
            "Formula_only": "pending independent fair control assessment",
            "extra_seeds": "pending selected optimizer stability assessment",
            "automatic_retry": False,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write(
            ROOT / "resume_queue_failure_20261026.json",
            {"type": type(exc).__name__, "message": str(exc), "automatic_retry": False},
        )
        raise
