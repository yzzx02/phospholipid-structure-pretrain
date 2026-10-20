"""Audit reuse of completed native jobs after the canonical-Polar validation repair."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.training.quality_v03 import polar_checks


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@lru_cache(maxsize=65536)
def checked(text, eos):
    return polar_checks(text, eos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    old, new = args.prior_root.resolve(), args.root.resolve()
    output = new / "prior_native_validation_audit.json"
    if output.exists() or (new / "gpu_monitor_epoch_private.json").exists():
        raise ValueError("Fresh revision audit and monitor epoch required")
    stop = read(old / "GPU_STOP.json")
    failed = old / stop["run"]
    host = read(failed / "host_monitor.json")
    stderr = (failed / "stderr.log").read_text()
    if (
        stop["reason"] != "child_failed"
        or host["new_windows_gpu_dump_count"]
        or host["display_reset_count"]
        or not host["checkpoints_unchanged"]
        or "KekulizeException" not in stderr
        or "LibraryValidationError" not in stderr
        or "CUDA error" in stderr
    ):
        raise ValueError(
            "This audited revision is only for the proven CPU-reproducible validator bug"
        )
    old_gate, new_gate = read(old / "cpu_gate.json"), read(new / "cpu_gate.json")
    if (
        not new_gate["passed"]
        or old_gate["cpu_checkpoint_audits"] != new_gate["cpu_checkpoint_audits"]
    ):
        raise ValueError("Frozen checkpoint/data contracts must remain identical")
    for name in (
        "models/polar_decoder.py",
        "polar_only_v04/constrained_decode.py",
        "polar_only_v04/runtime.py",
        "utils/attention.py",
    ):
        key = "src/structure_pretrain/" + name
        if old_gate["inference_source_sha256"][key] != new_gate["inference_source_sha256"][key]:
            raise ValueError("Decoder probabilities, constraints and runtime must remain unchanged")
    jobs, candidates, changes = [], 0, []
    for path in sorted(old.glob("*/host_monitor.json")):
        monitor = read(path)
        if (
            monitor["child_exit_code"]
            or monitor["stop_reason"]
            or monitor["new_windows_gpu_dump_count"]
            or monitor["display_reset_count"]
            or not monitor["checkpoints_unchanged"]
        ):
            continue
        folder = path.parent
        count = 0
        for line in (folder / "predictions_private.jsonl").read_text().splitlines():
            record = json.loads(line)
            for candidate in record["raw_candidates"]:
                current = checked(candidate["polar_smiles_internal"], candidate["eos"])
                if current["polar_valid"] != candidate["polar_valid"] or current.get(
                    "canonical_polar"
                ) != candidate.get("canonical_polar"):
                    changes.append({"job": folder.name, "cohort_index": record["cohort_index"]})
                count += 1
        candidates += count
        jobs.append(
            {
                "name": folder.name,
                "candidates_checked": count,
                "metrics_sha256": file_hash(folder / "metrics.json"),
                "host_monitor_sha256": file_hash(path),
                "predictions_sha256": file_hash(folder / "predictions_private.jsonl"),
            }
        )
    result = {
        "passed": not changes,
        "prior_root": str(old),
        "jobs": jobs,
        "candidates_checked": candidates,
        "classification_changes": changes,
        "validation_scope": "Reapply repaired polar_checks; canonical graphs and validity "
        "must match",
        "decoder_and_constraint_sources_unchanged": True,
        "prior_gate_sha256": file_hash(old / "cpu_gate.json"),
        "new_gate_sha256": file_hash(new / "cpu_gate.json"),
        "prior_stop_preserved": stop,
        "failed_partial_targets_excluded": len(
            (failed / "predictions_private.jsonl").read_text().splitlines()
        ),
        "prior_monitor_epoch_sha256": file_hash(old / "gpu_monitor_epoch_private.json"),
    }
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    if changes:
        raise ValueError("Completed results changed under the repair; reuse is forbidden")
    # Preserve the original observation interval, including the repair gap. The new
    # supervisor still checks all events/dump changes since that original UTC epoch.
    epoch = read(old / "gpu_monitor_epoch_private.json")
    epoch.update(
        cpu_gate_sha256=file_hash(new / "cpu_gate.json"), continued_from_prior_root=str(old)
    )
    with (new / "gpu_monitor_epoch_private.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(epoch, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "completed_jobs": len(jobs),
                "candidates_checked": candidates,
                "classification_changes": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
