"""Same FP/projector/set/decoder network with a whole-core formula boundary."""

import torch
from torch import nn
from torch.nn import functional as F

from structure_pretrain.models.fingerprint_encoder import FingerprintEncoder
from structure_pretrain.models.polar_decoder import PolarDecoder
from structure_pretrain.models.template_set_transformer import TemplateSetTransformer
from structure_pretrain.models.template_token_projector import TemplateTokenProjector


class GlycerolCoreModel(nn.Module):
    def __init__(self, config, tokenizer, statistics):
        super().__init__()
        self.structure_encoder = FingerprintEncoder(config.dropout)
        self.projector = TemplateTokenProjector(statistics, config.dropout)
        self.template_set = TemplateSetTransformer(config)
        self.polar_decoder = PolarDecoder(config, tokenizer)

    def encode_templates(self, batch):
        mask, core = batch["template_mask"], batch["target_core_formula"]
        if core.ndim != 2 or core.shape[-1] != 10 or not torch.isfinite(core).all():
            raise ValueError("Supply the neutral H-capped whole-core 10-element formula")
        if (core < 0).any() or not torch.equal(core, core.round()):
            raise ValueError("Formula must contain nonnegative integer element counts")
        if not mask.any(-1).all() or not torch.equal(
            (batch["template_formulas"] + batch["delta_formulas"])[mask],
            core[:, None].expand(-1, mask.shape[1], -1)[mask],
        ):
            raise ValueError("Missing template or formula/delta inconsistency")
        embedding = self.structure_encoder(batch["fingerprints"])
        structure = embedding.new_zeros((*mask.shape, 256))
        structure[mask] = embedding
        tokens = self.projector(
            structure, batch["template_formulas"], batch["delta_formulas"], mask
        )
        memory, _, padding, effective = self.template_set(tokens, mask)
        return {
            "template_memory": memory,
            "memory_padding_mask": padding,
            "effective_template_mask": effective,
            "target_core_formula": core,
        }

    def forward(self, batch):
        output = self.encode_templates(batch)
        output["polar_logits"] = self.polar_decoder(
            batch["target_polar_token_ids"][:, :-1],
            output["template_memory"],
            output["memory_padding_mask"],
        )
        return output


def loss(output, batch, pad_id):
    return F.cross_entropy(
        output["polar_logits"].transpose(1, 2),
        batch["target_polar_token_ids"][:, 1:],
        ignore_index=pad_id,
    )
