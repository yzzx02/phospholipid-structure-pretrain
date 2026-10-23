"""Verified rolling checkpoints confined to new V2 run directories."""

import gc
import json
import math
import os
import re
from pathlib import Path

import torch

from structure_pretrain.formal_v2.common import ROOT, digest, read, write


def no_reparse(path):
    return not path.is_symlink() and not getattr(path, "is_junction", lambda: False)()


def owned_bytes(root):
    total = 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if no_reparse(Path(folder) / name)]
        for name in files:
            p = Path(folder) / name
            if no_reparse(p):
                s = p.stat()
                if s.st_nlink == 1:
                    total += s.st_size
    return total


class CompactCheckpoints:
    """Retain best plus latest2, after atomic save + CPU reload + finite checks."""

    def __init__(self, run, *, experiment_root=ROOT, hard_limit_bytes=12 * 2**30):
        self.root = Path(experiment_root).resolve()
        self.run = Path(run).absolute()
        if not self.run.resolve().is_relative_to(self.root) or self.run.resolve() == self.root:
            raise ValueError("Checkpoint writes must stay in a dedicated new run")
        for p in [self.run, *list(self.run.parents)[: len(self.run.parts) - len(self.root.parts)]]:
            if p.exists() and not no_reparse(p):
                raise ValueError("No linked/junction checkpoint directories")
        self.folder = self.run / "checkpoint"
        self.folder.mkdir(parents=True, exist_ok=True)
        if not no_reparse(self.folder) or self.folder.resolve().parent != self.run.resolve():
            raise ValueError("Unsafe checkpoint directory")
        self.index = self.run / "checkpoint_retention.json"
        self.history = self.run / "checkpoint_history.jsonl"
        self.hard_limit_bytes = hard_limit_bytes

    def validate(self, temporary, state):
        loaded = torch.load(temporary, map_location="cpu", weights_only=False)
        for required in ("model", "optimizer", "scheduler", "rng", "progress", "contract"):
            if required not in loaded:
                raise ValueError("Incomplete checkpoint: " + required)
        if loaded["progress"] != state["progress"] or loaded["scheduler"] != state["scheduler"]:
            raise ValueError("Checkpoint progress/scheduler round-trip changed")
        for name, tensor in loaded["model"].items():
            if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonfinite checkpoint model: " + name)
        for values in loaded["optimizer"].get("state", {}).values():
            for tensor in values.values():
                if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
                    raise ValueError("Nonfinite checkpoint optimizer")
        del loaded
        gc.collect()

    def save(self, filename, state, *, best=False):
        if not re.fullmatch(r"step_\d{6,9}_(periodic|epoch_\d{3}|final)\.pt", filename):
            raise ValueError("Checkpoint name must be an exact managed filename")
        destination = self.folder / filename
        if destination.exists() or destination.with_suffix(".partial").exists():
            raise FileExistsError(destination)
        previous = (
            read(self.index)
            if self.index.exists()
            else {"latest": [], "best": None, "retained": {}}
        )
        present = {p.name for p in self.folder.glob("*.pt")}
        if present != set(previous["retained"]):
            raise ValueError("Unregistered or missing checkpoint; preserve all files and stop")
        for name, metadata in previous["retained"].items():
            if digest(self.folder / name) != metadata["sha256"]:
                raise ValueError("Previously retained checkpoint hash changed")
        if owned_bytes(self.root) + 256 * 2**20 > self.hard_limit_bytes:
            raise ValueError("New V2 owned-output storage budget exhausted; no silent deletion")
        temporary = destination.with_suffix(".partial")
        torch.save(state, temporary)
        self.validate(temporary, state)
        sha = digest(temporary)
        temporary.replace(destination)
        latest = ([filename] + previous["latest"])[:2]
        best_name = filename if best else previous["best"]
        keep = set(latest) | ({best_name} if best_name else set())
        entry = {
            "filename": filename,
            "path": str(destination.resolve()),
            "sha256": sha,
            "step": state["progress"]["step"],
            "epoch": state["progress"]["epoch"],
            "best_validation_loss": (
                state["progress"]["best_validation_loss"]
                if math.isfinite(state["progress"]["best_validation_loss"])
                else None
            ),
            "CPU_reload_verified": True,
        }
        retained = {**previous["retained"], filename: entry}
        canonical = {
            "schema": "new_v2_best_plus_latest2",
            "latest": latest,
            "best": best_name,
            "retained": {name: retained[name] for name in keep},
        }
        write(self.index, canonical)
        write(self.run / "last.json", entry)
        if best:
            write(self.run / "best.json", entry)
        with self.history.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "saved_verified", "best": best, **entry}) + "\n")
        # Delete ONLY registered obsolete files in this ordinary new checkpoint folder.
        for name, metadata in previous["retained"].items():
            if name in keep:
                continue
            obsolete = self.folder / name
            if (
                not re.fullmatch(r"step_\d{6,9}_(periodic|epoch_\d{3}|final)\.pt", name)
                or obsolete.resolve().parent != self.folder.resolve()
                or not no_reparse(obsolete)
                or digest(obsolete) != metadata["sha256"]
            ):
                raise ValueError("Unsafe/changed obsolete checkpoint; preserve and stop")
            obsolete.unlink()
            with self.history.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "event": "pruned_verified_obsolete",
                            "filename": name,
                            "sha256": metadata["sha256"],
                        }
                    )
                    + "\n"
                )
        if len(list(self.folder.glob("*.pt"))) > 3:
            raise ValueError("Unregistered extra checkpoint; preserve and stop")
        return entry
