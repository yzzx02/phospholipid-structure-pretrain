"""CPU-only ZIP64 export of v0.4 essentials and the two 3000-step checkpoints."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = "phospholipid-structure-pretrain"
PART_BYTES = 2 * 1024**3
BLOCK = 4 * 1024**2


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def windows_path(relative):
    for part in relative.parts:
        if re.search(r'[<>:"|?*\\\x00-\x1f]', part) or part.rstrip(" .") != part:
            raise ValueError(f"Windows-incompatible path: {relative}")
        if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", part.split(".")[0], re.I):
            raise ValueError(f"Windows reserved path: {relative}")


def emit(**record):
    print(json.dumps(record), flush=True)


def selected_paths(root):
    """Keep frozen runtime inputs; omit old runs and retrieval-build intermediates."""
    paths = set(
        Path(os.fsdecode(name))
        for name in subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
        if name
    )
    for directory in (
        "data/structure_library_v0.3",
        "data/private/pretraining_v03/rows",
        "data/private/pretraining_v04/fragments",
        "outputs/pretraining_v04/samples",
    ):
        for path in (root / directory).rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Inspect symlink before transfer: {path}")
            if path.is_file():
                paths.add(path.relative_to(root))
    names = (
        "asset_archive_hashes.json",
        "chain_labels.npy",
        "chain_vocab.json",
        "formulas.npy",
        "group_row_counts.npy",
        "group_splits.npy",
        "input_manifest.json",
        "label_audit.json",
        "labels_manifest.json",
        "polar_groups.npy",
        "polar_groups.parquet",
        "polar_tokenizer.json",
        "reconstruction_audit.json",
        "row_splits.npy",
        "split_audit.json",
        "tokenization_audit.json",
    )
    paths.update(Path("data/private/pretraining_v03") / name for name in names)
    for path in (root / "data/private/pretraining_v04").iterdir():
        if path.is_file() and not (
            path.name.startswith(("HC_eligible_", "ann_"))
            or path.name == "train_chainless_morgan.usearch"
        ):
            paths.add(path.relative_to(root))
    for encoder in ("fp", "gine"):
        paths.add(
            Path(f"outputs/pretraining_v04/runs/{encoder}_smoke_3000/checkpoint/step_003000.pt")
        )
    for name in (
        "outputs/pretraining_v04/GPU_STOP.json",
        "outputs/formula_constrained_v04_20261020/GPU_STOP.json",
        "outputs/formula_constrained_v04_20261020/gpu_safety_final.json",
    ):
        if (root / name).exists():
            paths.add(Path(name))
    for path in paths:
        if not (root / path).is_file():
            raise ValueError(f"Required frozen file missing: {path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh folder outside the repo")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if root.name != PROJECT:
        parser.error("Run from the canonical project root")
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error("Use a fresh output folder outside the project")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit code/reports before building a reproducible transfer")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    paths = selected_paths(root)
    seen = set()
    for path in paths:
        windows_path(path)
        key = str(path).casefold()
        if key in seen:
            raise ValueError(f"Windows case collision: {path}")
        seen.add(key)
    output.mkdir(parents=True)
    cache = output.parent / (output.name + "_build_cache")
    cache.mkdir(exist_ok=False)
    sources = [(str(path), root / path) for path in sorted(paths)]
    total = sum(path.stat().st_size for _, path in sources)
    emit(stage="hashing", files=len(sources), bytes=total, source_commit=commit)
    files, stats = [], {}
    for number, (name, path) in enumerate(sources, 1):
        info = path.stat()
        files.append({"path": name, "bytes": info.st_size, "sha256": sha256(path)})
        stats[name] = (info.st_size, info.st_mtime_ns)
        if number % 100 == 0:
            emit(stage="hashing", files=number, total_files=len(sources))
    manifest = {
        "schema": "phospholipid-transfer-v1",
        "source_commit": commit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": "native Windows; user's existing default CUDA environment",
        "file_count": len(files),
        "uncompressed_bytes": total,
        "checkpoints": sum(Path(f["path"]).suffix in (".pt", ".pth", ".ckpt") for f in files),
        "profile": "v0.4-essential-two-3000-step-checkpoints",
        "excluded": [
            "historical test/training outputs and intermediate checkpoints",
            "v0.3 train bank, obsolete retrieval caches and Morgan index",
            "v0.4 ANN construction intermediates",
            "Linux/ROCm environments and Git history",
        ],
        "files": files,
    }
    manifest_path = output / "PROJECT_TRANSFER_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    zip_path = cache / "phospholipid_v04_transfer.zip"
    started = time.monotonic()
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for number, (name, path) in enumerate(sources, 1):
            if (path.stat().st_size, path.stat().st_mtime_ns) != stats[name]:
                raise ValueError(f"Source changed during packaging: {name}")
            archive.write(path, f"{PROJECT}/{name}")
            if number % 100 == 0:
                emit(stage="ZIP64", files=number, total_files=len(sources))
        archive.write(manifest_path, f"{PROJECT}/{manifest_path.name}")
    emit(stage="verifying_full_archive", archive_bytes=zip_path.stat().st_size)
    with zipfile.ZipFile(zip_path) as archive:
        if len(archive.infolist()) != len(files) + 1:
            raise ValueError("Unexpected ZIP entries")
        for number, file in enumerate(files, 1):
            digest = hashlib.sha256()
            with archive.open(f"{PROJECT}/{file['path']}") as stream:
                for block in iter(lambda: stream.read(BLOCK), b""):
                    digest.update(block)
            if digest.hexdigest() != file["sha256"]:
                raise ValueError(f"ZIP roundtrip checksum mismatch: {file['path']}")
            if number % 100 == 0:
                emit(stage="verifying_full_archive", files=number, total_files=len(files))
    parts, index = [], 1
    emit(stage="splitting", part_max_bytes=PART_BYTES)
    with zip_path.open("rb") as source:
        while True:
            first = source.read(BLOCK)
            if not first:
                break
            part = output / f"phospholipid_v04_transfer.zip.{index:03d}"
            digest, written = hashlib.sha256(), 0
            with part.open("xb") as stream:
                block = first
                while block:
                    stream.write(block)
                    digest.update(block)
                    written += len(block)
                    if written == PART_BYTES:
                        break
                    block = source.read(min(BLOCK, PART_BYTES - written))
            parts.append({"name": part.name, "bytes": written, "sha256": digest.hexdigest()})
            emit(stage="part_ready", **parts[-1])
            index += 1
    shutil.copy2(root / "scripts/restore_transfer_windows.ps1", output / "RESTORE_WINDOWS.ps1")
    shutil.copy2(root / "docs/HANDOFF_RTX4070_WINDOWS.md", output / "READ_ME_FIRST.md")
    receipt = {
        "status": "verified_ready",
        "source_commit": commit,
        "file_count": len(files),
        "checkpoints": manifest["checkpoints"],
        "uncompressed_bytes": total,
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "manifest_sha256": sha256(manifest_path),
        "parts": parts,
        "full_ZIP_roundtrip_SHA256_verified": True,
        "environment_changed": False,
        "GPU_used_for_export": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "PACKAGE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    sums = output / "SHA256SUMS.txt"
    sums.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path != sums
        )
    )
    emit(stage="complete", **{k: v for k, v in receipt.items() if k != "parts"})


if __name__ == "__main__":
    main()
