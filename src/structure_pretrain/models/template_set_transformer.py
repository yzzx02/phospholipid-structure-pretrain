import torch
from torch import nn

from structure_pretrain.config import ModelConfig
from structure_pretrain.utils.masks import drop_templates


class TemplateSetTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.template_dropout = config.template_dropout
        self.tcls = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.set_heads,
            config.set_ffn_dim,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.set_layers, nn.LayerNorm(config.d_model), enable_nested_tensor=False
        )

    def forward(self, tokens, mask):
        effective = drop_templates(mask, self.template_dropout, self.training)
        tokens = tokens.masked_fill(~effective.unsqueeze(-1), 0)
        inputs = torch.cat((self.tcls.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        padding = torch.cat((torch.zeros_like(effective[:, :1]), ~effective), dim=1)
        memory = self.encoder(inputs, src_key_padding_mask=padding)
        # Also zero masked output rows: masked tokens cannot later become decoder memory.
        memory = memory.masked_fill(padding.unsqueeze(-1), 0)
        return memory, memory[:, 0], padding, effective
