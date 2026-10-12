import torch
from torch import nn

from structure_pretrain.utils.formula import FormulaStandardizer


class TemplateTokenProjector(nn.Module):
    def __init__(self, statistics: dict | None = None, dropout: float = 0.1):
        super().__init__()
        self.normalizer = FormulaStandardizer(statistics)
        self.mlp = nn.Sequential(
            nn.Linear(278, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

    def forward(self, structure, template_formula, delta, similarity, available, mask):
        template_z, delta_z = self.normalizer(template_formula, delta, available)
        joined = torch.cat(
            (
                structure,
                template_z,
                delta_z,
                similarity.unsqueeze(-1),
                available.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        if joined.shape[-1] != 278:
            raise ValueError("Expected 256+10+10+1+1=278 input dimensions")
        return self.mlp(joined).masked_fill(~mask.unsqueeze(-1), 0)
