import torch
from torch import nn

from structure_pretrain.config import ModelConfig
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer


class PolarDecoder(nn.Module):
    def __init__(self, config: ModelConfig, tokenizer: SmilesTokenizer):
        super().__init__()
        self.pad_id = tokenizer.pad_id
        self.max_length = config.max_polar_length
        self.bos_id, self.eos_id, self.unk_id = tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id
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
            raise ValueError(
                "Polar sequence exceeds audited configured length; truncation forbidden"
            )
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

    @torch.no_grad()
    def generate(self, memory, padding, beam_size=1, max_length=None, progress=None):
        """BOS-only autoregression. Finished hypotheses carry their actual EOS status.

        No target prefixes, forced EOS, chemistry repair, or formula filtering.
        Beam ranking uses mean log probability (length penalty=1).
        """
        limit = self.max_length if max_length is None else max_length
        if not 2 <= limit <= self.max_length or beam_size < 1:
            raise ValueError("Generation length/beam outside decoder contract")
        results = []
        for b in range(len(memory)):
            beams = [([self.bos_id], 0.0, False)]
            for step in range(limit - 1):
                if progress is not None:
                    progress(step)
                active = [beam for beam in beams if not beam[2]]
                if not active:
                    break
                ids = torch.tensor([beam[0] for beam in active], device=memory.device)
                logits = self(
                    ids,
                    memory[b : b + 1].expand(len(active), -1, -1),
                    padding[b : b + 1].expand(len(active), -1),
                )[:, -1].clone()
                logits[:, [self.pad_id, self.bos_id, self.unk_id]] = -torch.inf
                logp = logits.log_softmax(-1)
                values, choices = logp.topk(min(beam_size, logp.shape[-1] - 3))
                candidates = [beam for beam in beams if beam[2]]
                for i, (prefix, score, _) in enumerate(active):
                    for value, token in zip(values[i].tolist(), choices[i].tolist()):
                        candidates.append((prefix + [token], score + value, token == self.eos_id))
                beams = sorted(candidates, key=lambda x: x[1] / (len(x[0]) - 1), reverse=True)[
                    :beam_size
                ]
            results.append(
                [
                    {
                        "ids": ids,
                        "log_probability": score,
                        "eos": done,
                        "length_exhausted": not done,
                    }
                    for ids, score, done in beams
                ]
            )
        return results
