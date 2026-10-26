"""Small precomputed arrays; all bank structures are also target labels."""

from pathlib import Path

import numpy as np
import torch

from structure_pretrain.core_templates.common import ROOT, model_config, read, tokenizer
from structure_pretrain.core_templates.model import GlycerolCoreModel


def make_model(root=ROOT):
    return GlycerolCoreModel(
        model_config(), tokenizer(root), read(Path(root) / "data/formula_statistics.json")
    )


class CoreData:
    def __init__(self, root=ROOT, device="cpu"):
        self.root, self.device = Path(root), torch.device(device)
        folder = self.root / "data"
        self.manifest = read(folder / "manifest.json")
        self.tokenizer = tokenizer(root)
        self.lengths = np.load(folder / "token_lengths.npy")
        self.splits = np.load(folder / "splits.npy")
        self.bank_rows = np.load(folder / "bank_rows.npy")
        self.tokens = torch.as_tensor(
            np.load(folder / "tokens.npy").astype(np.int64), device=device
        )
        self.formulas = torch.as_tensor(
            np.load(folder / "formulas.npy").astype(np.float32), device=device
        )
        self.refs = torch.as_tensor(
            np.load(folder / "references.npy").astype(np.int64), device=device
        )
        self.bank_formula = self.formulas[torch.as_tensor(self.bank_rows, device=device)]
        self.bank_fp = torch.as_tensor(
            np.unpackbits(np.load(folder / "bank_packed.npy"), axis=-1, bitorder="little").astype(
                np.float32
            ),
            device=device,
        )

    def indices(self, split):
        return np.flatnonzero(self.splits == split)

    def batch(self, indices, k=3):
        indices = np.asarray(indices, dtype=np.int64)
        idx = torch.as_tensor(indices, device=self.device)
        refs = self.refs[idx, :k]
        mask, safe = refs >= 0, refs.clamp_min(0)
        formulas = self.bank_formula[safe].masked_fill(~mask[:, :, None], 0)
        delta = (self.formulas[idx, None] - formulas).masked_fill(~mask[:, :, None], 0)
        return {
            "fingerprints": self.bank_fp[safe][mask],
            "template_formulas": formulas,
            "delta_formulas": delta,
            "template_mask": mask,
            "target_core_formula": self.formulas[idx],
            "target_polar_token_ids": self.tokens[idx, : int(self.lengths[indices].max())],
        }
