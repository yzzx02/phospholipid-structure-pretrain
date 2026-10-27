"""One-shot publication of completed comparison aggregates, with strict private-data boundaries."""

import json
import subprocess
from pathlib import Path

from structure_pretrain.core_ablation.common import (
    ROOT,
    contract,
    digest,
    read,
    write,
)

FILES = [
    Path("docs/results/CORE_FORMULA_COMPARISON_20261027.md"),
    Path("docs/results/CORE_FORMULA_COMPARISON_20261027.json"),
]


def run(args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=120)


def private_fields_absent(value):
    if isinstance(value, dict):
        forbidden = {
            "target_id",
            "headgroup_id",
            "target_smiles",
            "polar_smiles",
            "original_smiles",
            "capped_core_smiles",
            "anchored_graph_key",
            "fingerprints",
            "target_polar_token_ids",
            "candidates",
        }
        if forbidden & value.keys():
            raise ValueError("Refuse publication of per-target/private fields")
        for item in value.values():
            private_fields_absent(item)
    elif isinstance(value, list):
        for item in value:
            private_fields_absent(item)


def publish():
    marker = ROOT / "PUBLICATION_DONE.json"
    if marker.exists():
        return read(marker)
    data = read(FILES[1])
    assert read(ROOT / "state.json")["status"] == data["state"]["status"] == "completed"
    assert data["formula_only_training"]["status"] == "completed"
    assert contract() == read(ROOT / "queue_started.json")["contract"]
    assert data["parameter_counts"] == {"template": 16479940, "formula_only": 16479940}
    best = read(ROOT / "formula_only_train_seed42/best.json")
    history = read(ROOT / "formula_only_train_seed42/validation_history.json")
    assert best["best_validation_loss"] == min(r["validation_token_CE"] for r in history)
    assert digest(best["path"]) == best["sha256"]
    assert (
        digest(data["configuration"]["template_checkpoint"])
        == data["template_epoch43_weight_sha256"]
    )
    audits = []
    for kind, modes in data["evaluations"].items():
        for mode, value in modes.items():
            assert value is not None and value["status"] == "completed"
            assert (
                value["targets"] == 9499
                and value["P"]["targets"] == 8225
                and value["nonP"]["targets"] == 1274
            )
            run_dir = ROOT / f"{kind}_full_{mode}"
            assert read(run_dir / "evaluation_report.json") == value
            saved = read(run_dir / "evaluation_contract.json")
            assert saved["contract"] == read(ROOT / "queue_started.json")["contract"]
            assert saved["raw_predictions_sha256"] == digest(
                run_dir / "raw_candidates_private.jsonl.gz"
            )
            monitor = read(run_dir / "host_monitor.json")
            assert monitor["child_exit_code"] == 0 and monitor["stop_reason"] is None
            expected = (
                best["sha256"] if kind == "formula_only" else data["template_epoch43_weight_sha256"]
            )
            assert saved["checkpoint_sha256"] == value["checkpoint_sha256"] == expected
            audits.append(
                {
                    "model": kind,
                    "mode": mode,
                    "targets": value["targets"],
                    "saved_candidate_stream_SHA_verified": True,
                }
            )
    assert len(audits) == 6 and len(data["paired_template_vs_formula"]) == 3
    private_fields_absent(data)
    assert run(["git", "branch", "--show-current"]).stdout.strip() == "main"
    assert (
        run(["git", "remote", "get-url", "origin"]).stdout.strip()
        == "https://github.com/yzzx02/phospholipid-structure-pretrain.git"
    )
    if run(["git", "diff", "--cached", "--name-only"]).stdout.strip():
        raise ValueError("Existing staged work: preserve it and leave publication pending")
    run(["git", "add", *[p.as_posix() for p in FILES]])
    changed = run(["git", "diff", "--cached", "--name-only"]).stdout.splitlines()
    if set(changed) - {p.as_posix() for p in FILES}:
        raise ValueError("Unexpected staged files: no commit or cleanup performed")
    run(["git", "diff", "--cached", "--check"])
    if changed:
        run(
            [
                "git",
                "commit",
                "-m",
                "Publish full9499 constrained generation and matched formula-only results",
            ]
        )
    sha = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    run(["git", "-c", "http.proxy=http://127.0.0.1:17891", "push", "origin", "main"])
    remote = run(
        ["git", "-c", "http.proxy=http://127.0.0.1:17891", "ls-remote", "origin", "refs/heads/main"]
    ).stdout.split()[0]
    assert remote == sha
    receipt = {
        "status": "published",
        "commit_sha": sha,
        "files_sha256": {p.as_posix(): digest(p) for p in FILES},
        "six_full_cohorts_verified": audits,
        "GPU_jobs_started": 0,
        "private_structures_weights_and_raw_predictions_uploaded": False,
    }
    write(marker, receipt)
    return receipt


if __name__ == "__main__":
    try:
        print(json.dumps(publish(), ensure_ascii=False), flush=True)
    except Exception as exc:
        write(
            ROOT / "PUBLICATION_PENDING.json",
            {
                "reason": type(exc).__name__ + ":" + str(exc),
                "numerical_results_preserved": True,
                "GPU_retry_attempted": False,
            },
        )
        raise
