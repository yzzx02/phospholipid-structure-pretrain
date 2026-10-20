"""Record the actual native CUDA environment and frozen transfer/checkpoint identities."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from structure_pretrain.library_builder.pipeline import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Environment receipt is protected")
    packages = (
        "torch",
        "torch-geometric",
        "rdkit",
        "numpy",
        "lightning",
        "omegaconf",
        "pyarrow",
        "pytest",
        "ruff",
        "usearch",
        "psutil",
    )
    workspace = Path.cwd().parent
    manifest = workspace / "phospholipid_rtx4070_transfer_20261020/PROJECT_TRANSFER_MANIFEST.json"
    transferred = json.loads(manifest.read_text())
    checkpoint_files = [f for f in transferred["files"] if f["path"].endswith("step_003000.pt")]
    checkpoints = {}
    for item in checkpoint_files:
        actual = file_hash(Path(item["path"]))
        if actual != item["sha256"]:
            raise ValueError("Checkpoint differs from verified transfer manifest")
        checkpoints[item["path"]] = actual
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "interpreter": sys.executable,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda_build": torch.version.cuda,
        "hip_build": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "GPU_total_bytes": torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else None,
        "RAM_total_bytes": psutil.virtual_memory().total,
        "logical_cpus": psutil.cpu_count(),
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "transfer_source_commit": transferred["source_commit"],
        "transfer_manifest_sha256": file_hash(manifest),
        "transfer_file_count_verified": transferred["file_count"],
        "checkpoints_match_transfer_manifest": True,
        "checkpoints_sha256": checkpoints,
        "training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
