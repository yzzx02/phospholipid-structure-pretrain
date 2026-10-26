"""Hashed, path-bounded cleanup of disposable caches; never traverse junctions."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path("G:/yzx/linzhiyuce")
PROJECT = WORKSPACE / "phospholipid-structure-pretrain"
ROOT = PROJECT / "outputs/glycerol_core_unified_v04_20261026"
V03 = PROJECT / "outputs/glycerol_core_collection_v03_20261026"
PIP = WORKSPACE / "cache/pip"
WHEEL = WORKSPACE / "cache/wheels/torch-2.9.1+cu128-cp312-cp312-win_amd64.whl"
WHEEL_SHA = "3a01f0b64c10a82d444d9fd06b3e8c567b1158b76b2764b8f51bfd8f535064b0"
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def ordinary_source_cache(path):
    relative = path.relative_to(PROJECT)
    if not CACHE_NAMES.intersection(relative.parts):
        return False
    if "__pycache__" in relative.parts:
        if path.suffix != ".pyc":
            return False
        try:
            return Path(importlib.util.source_from_cache(str(path))).is_file()
        except ValueError:
            return False
    return True


def permitted(path):
    if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return False
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(WORKSPACE.resolve()) or resolved != path.absolute():
        return False
    if path == WHEEL or path.is_relative_to(PIP):
        return True
    if path.is_relative_to(PROJECT):
        parts = path.relative_to(PROJECT).parts
        if any(p in (".git", "data", "environments", "restored_v04") for p in parts):
            return False
        if ordinary_source_cache(path):
            return True
    parts_root = V03 / "raw/element_recheck_parts"
    if path.parent == parts_root and path.name.startswith(("input_", "result_")):
        return path.name.endswith(".json.gz")
    if path.parent == V03 / "strict_core_attempt":
        return path.name in {
            "pending_source_or_core.parquet.part",
            "observed_source_mapping.jsonl.gz",
        }
    return False


def physical_files(folder, *, project_scan=False):
    stack = [folder]
    while stack:
        current = stack.pop()
        if not current.exists():
            continue
        for item in os.scandir(current):
            path = Path(item.path)
            info = path.stat(follow_symlinks=False)
            if info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                continue
            if item.is_dir(follow_symlinks=False):
                if project_scan and item.name in {".git", "data", "environments", "node_modules"}:
                    continue
                stack.append(path)
            elif item.is_file(follow_symlinks=False):
                yield path


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    targets = {}
    for path in physical_files(PIP):
        targets[path] = "pip_download_cache"
    for path in physical_files(PROJECT, project_scan=True):
        if ordinary_source_cache(path):
            targets[path] = "rebuildable_project_python_test_or_lint_cache"
    for path in physical_files(V03 / "raw/element_recheck_parts"):
        if permitted(path):
            targets[path] = "completed_transitional_recheck_parts_rebuildable_from_retained_sources"
    for path in physical_files(V03 / "strict_core_attempt"):
        if permitted(path):
            targets[path] = "superseded_incomplete_CPU_extraction_outputs"
    installed = importlib.metadata.version("torch")
    if WHEEL.exists():
        assert installed == "2.9.1+cu128"
        assert digest(WHEEL) == WHEEL_SHA
        targets[WHEEL] = "offline_installer_cache_installed_runtime_kept_and_version_hash_recorded"
    files = []
    for path, reason in sorted(targets.items()):
        assert permitted(path), str(path)
        info = path.stat()
        files.append(
            {
                "path": str(path),
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "sha256": WHEEL_SHA if path == WHEEL else digest(path),
                "reason": reason,
            }
        )
    plan = {
        "dry_run": True,
        "created_UTC": datetime.now(timezone.utc).isoformat(),
        "installed_torch_version": installed,
        "installed_runtime_root_kept": str(WORKSPACE / "environments/polar-v04-cu128"),
        "files": files,
        "logical_bytes": sum(f["bytes"] for f in files),
        "preserved": [
            "original195306_library",
            "all_checkpoints_and_predictions",
            "transfer_and_restored_trees",
            "current_V2_inputs_and_monitor_stop_locks",
            "complete_nonP_source_records_and_provenance",
            "all_manual_review_data",
        ],
    }
    (ROOT / "cache_cleanup_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"dry_run": True, "files": len(files), "logical_GB": plan["logical_bytes"] / 1e9}
        ),
        flush=True,
    )


def apply():
    manifest = json.loads((ROOT / "library_manifest_v04.json").read_text("utf-8"))
    assert manifest["main_count"] == 189864 and manifest["main_nonP_count"] == 25494
    plan = json.loads((ROOT / "cache_cleanup_plan.json").read_text("utf-8"))
    assert importlib.metadata.version("torch") == plan["installed_torch_version"]
    for name in ("main_core_library_v04", "standby_core_library_v04", "excluded_core_library_v04"):
        record = manifest["files"][name]
        assert digest(Path(record["path"])) == record["sha256"]
    before = shutil.disk_usage(WORKSPACE).free
    deleted, skipped = [], []
    for record in plan["files"]:
        path = Path(record["path"])
        if not path.exists():
            skipped.append({"path": str(path), "reason": "already_absent"})
            continue
        assert permitted(path)
        info = path.stat()
        if info.st_size != record["bytes"] or info.st_mtime_ns != record["mtime_ns"]:
            skipped.append({"path": str(path), "reason": "changed_after_dry_run"})
            continue
        assert digest(path) == record["sha256"]
        try:
            path.unlink()
            deleted.append(record)
        except PermissionError:
            skipped.append({"path": str(path), "reason": "in_use_or_access_denied"})
    result = {
        "applied": True,
        "deleted_files": len(deleted),
        "logical_bytes_removed": sum(r["bytes"] for r in deleted),
        "disk_free_before": before,
        "disk_free_after": shutil.disk_usage(WORKSPACE).free,
        "files": deleted,
        "skipped": skipped,
        "preserved": plan["preserved"],
        "installed_torch_version": plan["installed_torch_version"],
        "offline_installer_receipt": {
            "version": "2.9.1+cu128",
            "wheel_SHA256": WHEEL_SHA,
            "recovery_constraints": "constraints-cuda-v04.txt",
        },
    }
    (ROOT / "cache_cleanup_receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "applied": True,
                "deleted_files": len(deleted),
                "logical_GB_removed": result["logical_bytes_removed"] / 1e9,
                "skipped": len(skipped),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    apply() if args.apply else prepare()
