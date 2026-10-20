"""Read-only physical-tree inventory and hashed cache-only cleanup dry run."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path("G:/yzx/linzhiyuce")
ROOT = (
    WORKSPACE / "phospholipid-structure-pretrain/outputs/polar_similarity_eval_v04_20261020/storage"
)
CACHES = {"__pycache__", ".pytest_cache", ".ruff_cache"}
FULL_WHEEL = WORKSPACE / "cache/wheels/torch-2.9.1+cu128-cp312-cp312-win_amd64.whl"
WHEEL_SHA = "3a01f0b64c10a82d444d9fd06b3e8c567b1158b76b2764b8f51bfd8f535064b0"


def digest(path, limit=None):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while limit is None or limit > 0:
            block = f.read(4 * 1024 * 1024 if limit is None else min(limit, 4 * 1024 * 1024))
            if not block:
                break
            h.update(block)
            if limit is not None:
                limit -= len(block)
    return h.hexdigest()


def category(path):
    parts = path.relative_to(WORKSPACE).parts
    lower = path.name.lower()
    if parts[0] == "phospholipid_rtx4070_transfer_20261020":
        return "protected_transfer_archive_and_receipts"
    if parts[0] == "restored_v04":
        return "protected_frozen_restore_checkpoints_data_and_audits"
    if any(p in CACHES for p in parts):
        return "ordinary_python_or_test_cache"
    if path.suffix.lower() in (".pt", ".pth", ".ckpt"):
        return "checkpoint_keep_manual_review_only"
    if "predictions" in lower or "raw_beams" in lower:
        return "prediction_keep_evaluation_or_manual_review"
    if any(s in lower for s in ("audit", "manifest", "contract", "gpu_stop", "monitor", "receipt")):
        return "protected_audit_contract_monitor_or_stop"
    if parts[:2] == ("cache", "tmp"):
        return "temporary_download_or_export_review"
    if parts[:2] == ("cache", "wheels"):
        return "retained_offline_install_wheel"
    if parts[0] == "cache":
        return "download_cache_retained_for_reproducibility"
    if parts[0] == "environments":
        return "installed_runtime_retained"
    if "outputs" in parts:
        return "experiment_output_retained"
    return "source_or_version_history_retained"


def inventory():
    files, links, skipped = [], [], []
    stack = [WORKSPACE]
    seen = set()
    while stack:
        folder = stack.pop()
        for entry in os.scandir(folder):
            path = Path(entry.path)
            # DirEntry's cached Win32 metadata may return inode=0 for every file.
            s = path.stat(follow_symlinks=False)
            if s.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                links.append(
                    {
                        "path": str(path),
                        "target": str(path.resolve()),
                        "reason": "not traversed; target counted through physical tree",
                    }
                )
                continue
            if entry.is_dir(follow_symlinks=False):
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                inode = (s.st_dev, s.st_ino)
                if s.st_ino and inode in seen:
                    skipped.append(str(path))
                    continue
                seen.add(inode)
                files.append(
                    {
                        "path": str(path),
                        "bytes": s.st_size,
                        "mtime_ns": s.st_mtime_ns,
                        "category": category(path),
                    }
                )
    return files, links, skipped


def prepare(files, output):
    groups = defaultdict(list)
    for item in files:
        if item["category"] != "ordinary_python_or_test_cache":
            continue
        path = Path(item["path"])
        cache = next(p for p in (path, *path.parents) if p.name in CACHES)
        groups[str(cache)].append(item)
    targets = []
    for folder, members in sorted(groups.items()):
        if Path(folder).name == "__pycache__" and any(
            Path(f["path"]).suffix != ".pyc" for f in members
        ):
            continue
        targets.append(
            {
                "path": folder,
                "kind": "cache_directory",
                "reason": Path(folder).name,
                "files": [{**f, "sha256": digest(Path(f["path"]))} for f in members],
            }
        )
    full_valid = zipfile.is_zipfile(FULL_WHEEL) and digest(FULL_WHEEL) == WHEEL_SHA
    assert full_valid, "Retained full Torch wheel must match its historical verified SHA"
    partials = []
    for item in files:
        path = Path(item["path"])
        if item["category"] != "temporary_download_or_export_review" or path.suffix != ".whl":
            continue
        if not path.parent.name.startswith("pip-unpack-") or zipfile.is_zipfile(path):
            continue
        if path.name != FULL_WHEEL.name or item["bytes"] >= FULL_WHEEL.stat().st_size:
            continue
        sha = digest(path)
        evidence = {
            "incomplete_zip": True,
            "retained_complete_wheel_sha256": WHEEL_SHA,
            "identical_prefix_of_retained_wheel": sha == digest(FULL_WHEEL, item["bytes"]),
        }
        targets.append(
            {
                "path": str(path),
                "kind": "abandoned_partial_wheel",
                "reason": evidence,
                "files": [{**item, "sha256": sha}],
            }
        )
        partials.append(evidence)
    result = {
        "dry_run": True,
        "workspace": str(WORKSPACE.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "targets": targets,
        "target_files": sum(len(t["files"]) for t in targets),
        "logical_bytes": sum(f["bytes"] for t in targets for f in t["files"]),
        "full_torch_wheel_preserved": {"path": str(FULL_WHEEL), "sha256": WHEEL_SHA},
        "no_checkpoint_prediction_audit_or_data_delete_candidates": True,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("dry_run", "target_files", "logical_bytes")}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="initial")
    p.add_argument("--prepare-cleanup", action="store_true")
    args = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    files, links, duplicate_links = inventory()
    categories, directories = Counter(), Counter()
    for f in files:
        categories[f["category"]] += f["bytes"]
        relative = Path(f["path"]).relative_to(WORKSPACE)
        prefix = Path(*relative.parts[: min(3, len(relative.parts) - 1)])
        directories[str(prefix)] += f["bytes"]
    result = {
        "label": args.label,
        "utc": datetime.now(timezone.utc).isoformat(),
        "physical_tree_unique_regular_files": len(files),
        "logical_bytes": sum(categories.values()),
        "drive_free_bytes": shutil.disk_usage(WORKSPACE).free,
        "category_bytes": dict(categories.most_common()),
        "directory_bytes": dict(directories.most_common()),
        "junctions_not_double_counted": links,
        "hardlink_duplicates_skipped": duplicate_links,
        "largest_files": sorted(files, key=lambda f: -f["bytes"])[:40],
    }
    (ROOT / f"inventory_{args.label}_private.json").write_text(json.dumps(files, indent=2) + "\n")
    (ROOT / f"summary_{args.label}.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.prepare_cleanup:
        prepare(files, ROOT / "cleanup_dry_run_private.json")
    print(json.dumps(result["category_bytes"], indent=2))


if __name__ == "__main__":
    main()
