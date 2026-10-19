import torch
from torch import nn

from structure_pretrain.utils.formula import FormulaStandardizer


class TemplateTokenProjector(nn.Module):
    def __init__(self, statistics: dict | None = None, dropout: float = 0.1):
        super().__init__()
        self.normalizer = FormulaStandardizer(statistics)
        self.formula_mlp = nn.Sequential(
            nn.Linear(20, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(384, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

    def forward(self, structure, template_formula, delta, mask):
        if (
            not torch.isfinite(template_formula[mask]).all()
            or not torch.isfinite(delta[mask]).all()
        ):
            raise ValueError("Every real template must have finite mandatory formula/delta")
        if (template_formula[mask] < 0).any() or (template_formula[mask].sum(-1) <= 0).any():
            raise ValueError("Missing or invalid real-template neutral formula")
        template_z, delta_z = self.normalizer(template_formula, delta)
        formula = self.formula_mlp(torch.cat((template_z, delta_z), dim=-1))
        joined = torch.cat((structure, formula), dim=-1)
        if joined.shape[-1] != 384:
            raise ValueError("Expected structure 256D + formula 128D")
        return self.fusion_mlp(joined).masked_fill(~mask.unsqueeze(-1), 0)
