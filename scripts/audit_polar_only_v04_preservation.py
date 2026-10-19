"""Verify pre-v0.4 code/outputs and all320 transferred frozen-data files unchanged."""

import json
from pathlib import Path

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.prepare import dump


def main():
    root = Path("outputs/pretraining_v04")
    protected = read_json(root / "protected_v03_hashes_private.json")
    transfer = read_json("TRANSFER_V03_MANIFEST.json")
    problems = []
    checkpoints = 0
    for name, expected in protected.items():
        path = Path(name)
        if not path.exists() or file_hash(path) != expected:
            problems.append(name)
        checkpoints += path.suffix in (".pt", ".pth", ".ckpt")
    for entry in transfer["files"]:
        path = Path(entry["file"])
        if (
            not path.exists()
            or path.stat().st_size != entry["bytes"]
            or file_hash(path) != entry["sha256"]
        ):
            problems.append(entry["file"])
    report = {
        "status": "passed" if not problems else "failed",
        "protected_preexisting_code_and_output_files": len(protected),
        "protected_preexisting_checkpoints": checkpoints,
        "frozen_transfer_files": len(transfer["files"]),
        "frozen_transfer_bytes": transfer["uncompressed_data_bytes"],
        "modified_missing_or_mismatched": len(problems),
        "README_intentionally_excluded_for_new_v04_links": True,
        "old_v03_code_library_split_indexes_reports_outputs_checkpoints_unchanged": not problems,
    }
    dump(root / "preservation_audit.json", report)
    print(json.dumps(report))
    if problems:
        dump(root / "preservation_failures_private.json", problems)
        raise RuntimeError("Preexisting v0.3 assets differ; preserve evidence and diagnose")


if __name__ == "__main__":
    main()
