"""One-shot aggregate/figure publication, after all four validation cohorts finish."""

import subprocess
from pathlib import Path

from structure_pretrain.core_lr_study.common import (
    ROOT,
    contract,
    digest,
    original_run,
    read,
    write,
)

STEM = "docs/results/CORE_LR_SCHEDULE_STUDY_20261028"


def run(args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=120)


def assert_public(value):
    if isinstance(value, dict):
        forbidden = {
            "target_id",
            "headgroup_id",
            "target_smiles",
            "polar_smiles",
            "original_smiles",
            "anchored_graph_key",
            "candidates",
            "fingerprints",
            "target_polar_token_ids",
        }
        assert not forbidden.intersection(value)
        for item in value.values():
            assert_public(item)
    elif isinstance(value, list):
        for item in value:
            assert_public(item)


def publish():
    marker = ROOT / "PUBLICATION_DONE.json"
    if marker.exists():
        return read(marker)
    data = read(STEM + ".json")
    assert read(ROOT / "state.json")["status"] == data["status"] == "completed"
    assert data["common_window_has50_real_values_each"] and not data["test_used_for_tuning"]
    binding = contract()
    assert binding == read(ROOT / "queue_started.json")["contract"]
    shared = []
    for exp in ("E1", "E2", "E3", "E4"):
        run_dir = ROOT / f"{exp}_seed42"
        history = read(run_dir / "validation_history.json")
        assert 50 <= len(history) <= 80
        assert history == data["histories"][exp]
        pointer = read(run_dir / "best.json")
        assert digest(pointer["path"]) == pointer["sha256"]
        assert pointer["best_validation_loss"] == min(r["validation_token_CE"] for r in history)
        assert read(run_dir / "training_report.json")["status"] == "completed"
        if exp != "E3":
            shared.append(read(run_dir / "run_contract.json")["initial_parameter_sha256"])
        directory = ROOT / f"{exp}_validation_beam32"
        result = read(directory / "evaluation_report.json")
        assert result == data["evaluations"][exp]
        assert result["targets"] == 9493 and result["beam_size"] == 32
        assert result["P"]["targets"] + result["nonP"]["targets"] == 9493
        assert result["checkpoint_sha256"] == pointer["sha256"]
        evidence = read(directory / "evaluation_contract.json")
        assert evidence["contract"] == binding
        assert evidence["raw_predictions_sha256"] == digest(
            directory / "raw_candidates_private.jsonl.gz"
        )
        for folder in (run_dir, directory):
            monitor = read(folder / "host_monitor.json")
            assert monitor["child_exit_code"] == 0 and monitor["stop_reason"] is None
            assert monitor["protected_checkpoints_unchanged"]
    assert len(set(shared)) == 1 and shared[0] is not None
    assert (
        digest(read(original_run() / "best.json")["path"])
        == data["original_E3"]["best_weight_sha256"]
    )
    assert (
        digest(read(original_run() / "last.json")["path"])
        == data["original_E3"]["epoch48_weight_sha256"]
    )
    assert read(original_run() / "validation_history.json") == read(
        ROOT / "data/baseline_E3_history.json"
    )
    assert_public(data)
    files = [Path(STEM + ".md"), Path(STEM + ".json")]
    for name, sha in data["figures_sha256"].items():
        p = Path(name)
        assert p.parent == Path("docs/results/figures/core_lr_schedule_study_20261028")
        assert p.suffix in (".png", ".svg") and digest(p) == sha
        files.append(p)
    assert len(files) == 10
    assert run(["git", "branch", "--show-current"]).stdout.strip() == "main"
    assert (
        run(["git", "remote", "get-url", "origin"]).stdout.strip()
        == "https://github.com/yzzx02/phospholipid-structure-pretrain.git"
    )
    if run(["git", "diff", "--cached", "--name-only"]).stdout.strip():
        raise ValueError("Existing staged work preserved; publication pending")
    run(["git", "add", *[p.as_posix() for p in files]])
    changed = run(["git", "diff", "--cached", "--name-only"]).stdout.splitlines()
    assert not set(changed) - {p.as_posix() for p in files}
    run(["git", "diff", "--cached", "--check"])
    if changed:
        run(
            [
                "git",
                "commit",
                "-m",
                "Report four learning-rate schedules and validation Beam32 comparison",
            ]
        )
    sha = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    run(["git", "-c", "http.proxy=http://127.0.0.1:17891", "push", "origin", "main"])
    remote = run(
        ["git", "-c", "http.proxy=http://127.0.0.1:17891", "ls-remote", "origin", "refs/heads/main"]
    )
    assert remote.stdout.split()[0] == sha
    receipt = {
        "status": "published",
        "commit": sha,
        "files": [p.as_posix() for p in files],
        "private_data_published": False,
        "four_validation_cohorts": 9493,
        "test_used": False,
    }
    write(marker, receipt)
    return receipt


if __name__ == "__main__":
    try:
        print(publish(), flush=True)
    except Exception as exc:
        write(
            ROOT / "PUBLICATION_PENDING.json",
            {"status": "pending", "reason": str(exc), "GPU_retry": False},
        )
        raise
