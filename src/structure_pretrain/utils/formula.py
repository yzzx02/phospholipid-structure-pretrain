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
        # Welford parallel merge: bounded memory independent of pair count.
        total, mean, m2 = (
            0,
            torch.zeros(20, dtype=torch.float64),
            torch.zeros(20, dtype=torch.float64),
        )
        for item in examples:
            mask = item["template_mask"].bool()
            data = torch.cat((item["template_formulas"][mask], item["delta_formulas"][mask]), -1)
            data = data.reshape(-1, 20).double()
            if not len(data):
                continue
            if not torch.isfinite(data).all():
                raise ValueError("Nonfinite mandatory formula in training pairs")
            size = len(data)
            other = data.mean(0)
            diff = other - mean
            m2 += ((data - other) ** 2).sum(0) + diff**2 * total * size / (total + size)
            mean += diff * size / (total + size)
            total += size
        if not total:
            raise ValueError("Need at least one legal training template pair")
        std = (m2 / total).sqrt()
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        return {
            "template_mean": mean[:10].tolist(),
            "template_std": std[:10].tolist(),
            "delta_mean": mean[10:].tolist(),
            "delta_std": std[10:].tolist(),
            "pair_count": total,
        }

    def forward(self, template: torch.Tensor, delta: torch.Tensor):
        template_z = (template - self.template_mean) / self.template_std
        delta_z = (delta - self.delta_mean) / self.delta_std
        return template_z, delta_z
