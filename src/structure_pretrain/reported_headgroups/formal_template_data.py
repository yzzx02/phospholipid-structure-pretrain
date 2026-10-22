"""Independent fixed-bank data for final structure-conditioned generation training."""

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from structure_pretrain.config import ModelConfig
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.polar_only_v04.model import PolarOnlyModel

ROOT = Path("outputs/template_guided_fp5000_20261022_r1")
CONFIG = Path("configs/template_guided_fp5000.json")
FEATURES = Path("data/private/pretraining_v04")
V03 = Path("data/private/pretraining_v03")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def config():
    value = read(CONFIG)
    if (
        value["main_template_pool_size"] != 5000
        or value["precision"] != "fp32"
        or value["attention"] != "math"
        or value["tf32"]
        or value["checkpoint_selection"] != "minimum_validation_token_cross_entropy"
        or value["early_stopping_patience_epochs"] != 5
    ):
        raise ValueError("Formal paper-aligned configuration contract changed")
    return value


def model_config():
    base = OmegaConf.load(config()["base_model_configuration"])
    merged = OmegaConf.merge(OmegaConf.structured(ModelConfig()), base.model)
    return OmegaConf.to_object(merged)


def source_hashes():
    paths = [
        CONFIG,
        Path(config()["base_model_configuration"]),
        Path("scripts/prepare_template_guided_fp5000.py"),
        Path("train_template_guided_fp5000.py"),
        Path("scripts/gate_template_guided_fp5000.py"),
        Path("scripts/supervise_template_guided_fp5000.py"),
        Path("src/structure_pretrain/reported_headgroups/formal_template_data.py"),
        Path("src/structure_pretrain/polar_only_v04/model.py"),
        Path("src/structure_pretrain/polar_only_v04/chemistry.py"),
        Path("src/structure_pretrain/utils/attention.py"),
        Path("src/structure_pretrain/utils/masks.py"),
        Path("src/structure_pretrain/utils/formula.py"),
        Path("src/structure_pretrain/data/smiles_tokenizer.py"),
    ]
    paths += sorted(Path("src/structure_pretrain/models").glob("*.py"))
    return {p.as_posix(): digest(p) for p in paths}


def contract(root=ROOT):
    data = read(Path(root) / "data/manifest.json")
    return {
        "schema": config()["schema"],
        "configuration_sha256": digest(CONFIG),
        "data_manifest_sha256": digest(Path(root) / "data/manifest.json"),
        "data_assets_sha256": data["data_assets_sha256"],
        "source_sha256": source_hashes(),
        "model": asdict(model_config()),
        "runtime_versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "torch",
                    "torch-geometric",
                    "numpy",
                    "rdkit",
                    "scipy",
                    "usearch",
                    "omegaconf",
                )
            },
        },
    }


def verify_data(root=ROOT):
    manifest = read(Path(root) / "data/manifest.json")
    if manifest["status"] != "passed":
        raise ValueError("Prepared data audit must pass")
    for filename, expected in manifest["data_assets_sha256"].items():
        if digest(Path(root) / "data" / filename) != expected:
            raise ValueError("Formal data changed: " + filename)
    for path, expected in manifest["frozen_input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Frozen source changed: " + path)
    return manifest


def make_model(root=ROOT):
    tokenizer = SmilesTokenizer(read(V03 / "polar_tokenizer.json")["tokens"])
    return PolarOnlyModel(
        model_config(),
        tokenizer,
        read(Path(root) / "data/formula_statistics.json"),
        np.load(FEATURES / "chain_counts.npy"),
        np.load(FEATURES / "scaffold_counts.npy"),
    )


class CoreData:
    """Small immutable arrays staged once; no per-sample RDKit/FP recomputation."""

    def __init__(self, root=ROOT, device="cpu"):
        self.root, self.device = Path(root), torch.device(device)
        folder = self.root / "data"
        self.tokenizer = SmilesTokenizer(read(V03 / "polar_tokenizer.json")["tokens"])
        self.lengths = np.load(folder / "token_lengths.npy")
        self.splits = np.load(folder / "splits.npy")
        self.tokens = torch.as_tensor(
            np.load(folder / "tokens.npy").astype(np.int64), device=device
        )
        self.full = torch.as_tensor(
            np.load(folder / "full_formulas.npy").astype(np.float32), device=device
        )
        self.chains = torch.as_tensor(
            np.load(folder / "chain_labels.npy").astype(np.int64), device=device
        )
        self.polar = torch.as_tensor(
            np.load(folder / "polar_formulas.npy").astype(np.float32), device=device
        )
        self.refs = torch.as_tensor(
            np.load(folder / "references.npy").astype(np.int64), device=device
        )
        self.occupancy = torch.as_tensor(
            np.load(folder / "occupancy.npy").astype(np.int64), device=device
        )
        packed = np.load(folder / "bank_packed.npy")
        self.bank_fp = torch.as_tensor(
            np.unpackbits(packed, axis=-1, bitorder="little").astype(np.float32), device=device
        )
        self.bank_formula = torch.as_tensor(
            np.load(folder / "bank_formulas.npy").astype(np.float32), device=device
        )

    def indices(self, split):
        return np.flatnonzero(self.splits == split)

    def batch(self, indices, k=3):
        indices = np.asarray(indices, dtype=np.int64)
        width = int(self.lengths[indices].max())
        tensor = torch.as_tensor(indices, device=self.device)
        refs = self.refs[tensor, self.occupancy[tensor], :k]
        mask = refs >= 0
        safe = refs.clamp_min(0)
        form = self.bank_formula[safe]
        delta = self.polar[tensor, None] - form
        form = form.masked_fill(~mask[:, :, None], 0)
        delta = delta.masked_fill(~mask[:, :, None], 0)
        fp = self.bank_fp[self.occupancy[tensor, None], safe][mask]
        return {
            "fingerprints": fp,
            "template_formulas": form,
            "delta_formulas": delta,
            "template_mask": mask,
            "target_polar_token_ids": self.tokens[tensor, :width],
            "target_full_formula": self.full[tensor],
            "given_chain_tokens": self.chains[tensor],
        }


def epoch_batches(indices, lengths, batch_size, seed, epoch, megabatch_multiplier=50):
    """Deterministic one-core-per-epoch order, with less padding and exact resume."""
    random = np.random.default_rng(seed + 1000003 * epoch)
    order = random.permutation(indices)
    batches = []
    mega = batch_size * megabatch_multiplier
    for start in range(0, len(order), mega):
        chunk = order[start : start + mega]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        parts = [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
        batches.extend(parts[i] for i in random.permutation(len(parts)))
    return batches


def update_selection(best_loss, bad_epochs, validation_loss):
    """The only scientific checkpoint selector: strictly lower validation loss."""
    if not np.isfinite(validation_loss):
        raise ValueError("Nonfinite full-validation loss")
    if validation_loss < best_loss:
        return float(validation_loss), 0, True
    return float(best_loss), bad_epochs + 1, False
