"""Formula-only batches never open retrieval, fingerprint or reference-mask arrays."""

from pathlib import Path

import numpy as np
import torch

from structure_pretrain.core_ablation.common import ROOT, model_config, read, tokenizer
from structure_pretrain.core_ablation.model import FormulaOnlyModel


class FormulaData:
    def __init__(self, root=ROOT, device="cpu"):
        self.root, self.device = Path(root), torch.device(device)
        folder = self.root / "data"
        self.manifest = read(folder / "manifest.json")
        self.tokenizer = tokenizer()
        self.lengths = np.load(folder / "token_lengths.npy")
        self.splits = np.load(folder / "splits.npy")
        self.tokens = torch.as_tensor(
            np.load(folder / "tokens.npy").astype(np.int64), device=device
        )
        self.formulas = torch.as_tensor(
            np.load(folder / "formulas.npy").astype(np.float32), device=device
        )

    def indices(self, split):
        return np.flatnonzero(self.splits == split)

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        idx = torch.as_tensor(indices, device=self.device)
        return {
            "target_core_formula": self.formulas[idx],
            "target_polar_token_ids": self.tokens[idx, : int(self.lengths[indices].max())],
        }


def make_model(root=ROOT):
    return FormulaOnlyModel(
        model_config(), tokenizer(), read(Path(root) / "data/formula_statistics.json")
    )
