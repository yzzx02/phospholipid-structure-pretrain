"""Supplied neutral element counts only; no formula inference from structure or spectra."""

import math
from collections.abc import Iterable

import torch
from torch import nn

ELEMENTS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")


def validate_formula(counts: tuple[int, ...] | list[int]) -> None:
    if len(counts) != 10 or any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in counts
    ):
        raise ValueError("Supply 10 nonnegative integer neutral atom counts in ELEMENTS order")


class FormulaStandardizer(nn.Module):
    """Statistics fitted on valid TRAIN retrieval pairs and frozen in model buffers."""

    def __init__(self, statistics: dict | None = None):
        super().__init__()
        statistics = statistics or {}
        for name in ("template_mean", "template_std", "delta_mean", "delta_std"):
            default = [1.0 if name.endswith("std") else 0.0] * 10
            values = statistics.get(name, default)
            if len(values) != 10 or not all(math.isfinite(x) for x in values):
                raise ValueError("Formula statistics must be finite 10D vectors")
            if name.endswith("std") and any(x <= 0 for x in values):
                raise ValueError("Standard deviations must be positive")
            value = torch.tensor(values, dtype=torch.float32)
            self.register_buffer(name, value)

    @staticmethod
    def fit(examples: Iterable[dict]) -> dict:
        templates, deltas = [], []
        for item in examples:
            mask = item["template_mask"].bool()
            templates.append(item["template_formulas"][mask])
            available = mask & item["formula_available"].bool()
            deltas.append(item["delta_formulas"][available])

        def stats(parts):
            data = torch.cat(parts)
            if data.numel() == 0:
                raise ValueError("Need at least one available training formula to fit statistics")
            mean, std = data.mean(0), data.std(0, unbiased=False)
            # Constant elements use scale=1; no division by zero or tiny scale amplification.
            return mean.tolist(), torch.where(std > 1e-6, std, torch.ones_like(std)).tolist()

        tm, ts = stats(templates)
        dm, ds = stats(deltas)
        return {"template_mean": tm, "template_std": ts, "delta_mean": dm, "delta_std": ds}

    def forward(self, template: torch.Tensor, delta: torch.Tensor, available: torch.Tensor):
        template_z = (template - self.template_mean) / self.template_std
        delta_z = (delta - self.delta_mean) / self.delta_std
        # Zero AFTER standardization; missing query counts cannot leak through nonzero mean.
        delta_z = torch.where(available.bool().unsqueeze(-1), delta_z, torch.zeros_like(delta_z))
        return template_z, delta_z
