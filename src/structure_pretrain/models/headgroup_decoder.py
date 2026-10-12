import torch
from torch import nn

from structure_pretrain.config import ModelConfig
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer


class HeadgroupDecoder(nn.Module):
    def __init__(self, config: ModelConfig, tokenizer: SmilesTokenizer):
        super().__init__()
        self.pad_id = tokenizer.pad_id
        self.max_length = config.max_headgroup_length
        self.embedding = nn.Embedding(len(tokenizer), config.d_model, padding_idx=self.pad_id)
        self.positions = nn.Embedding(self.max_length, config.d_model)
        layer = nn.TransformerDecoderLayer(
            config.d_model,
            config.decoder_heads,
            config.decoder_ffn_dim,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, config.decoder_layers, nn.LayerNorm(config.d_model)
        )
        self.output = nn.Linear(config.d_model, len(tokenizer))

    def forward(self, input_ids, template_memory, memory_padding_mask):
        length = input_ids.shape[1]
        if not 1 <= length <= self.max_length:
            raise ValueError("Headgroup sequence exceeds configured length")
        positions = torch.arange(length, device=input_ids.device)
        state = self.embedding(input_ids) + self.positions(positions)[None]
        causal = torch.ones(length, length, device=input_ids.device, dtype=torch.bool).triu(1)
        decoded = self.decoder(
            state,
            template_memory,
            tgt_mask=causal,
            tgt_key_padding_mask=input_ids.eq(self.pad_id),
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output(decoded)
