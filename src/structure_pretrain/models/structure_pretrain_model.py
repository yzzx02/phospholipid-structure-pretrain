import torch
from torch import nn

from structure_pretrain.config import ModelConfig
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer

from .chain_head import ChainHead
from .fingerprint_encoder import FingerprintEncoder
from .graph_encoder import GraphEncoder
from .polar_decoder import PolarDecoder
from .template_set_transformer import TemplateSetTransformer
from .template_token_projector import TemplateTokenProjector


class StructurePretrainModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        chain_vocab_size: int,
        tokenizer: SmilesTokenizer,
        formula_statistics: dict | None = None,
    ):
        super().__init__()
        if (config.structure_dim, config.formula_dim, config.d_model) != (256, 128, 512):
            raise ValueError("v0.3 structure/formula/fusion dimensions must be 256/128/512")
        self.encoder_kind = config.encoder
        if config.encoder == "graph":
            self.structure_encoder = GraphEncoder(config.graph, config.dropout)
        elif config.encoder == "fingerprint":
            self.structure_encoder = FingerprintEncoder(config.dropout)
        else:
            raise ValueError("Unknown encoder")
        self.projector = TemplateTokenProjector(formula_statistics, config.dropout)
        self.template_set = TemplateSetTransformer(config)
        self.chain_head = ChainHead(chain_vocab_size, config.d_model)
        self.polar_decoder = PolarDecoder(config, tokenizer)

    def encode_templates(self, batch: dict):
        mask = batch["template_mask"]
        query = batch.get("target_formula")
        if (
            query is None
            or query.shape != (mask.shape[0], 10)
            or not torch.isfinite(query).all()
            or (query < 0).any()
            or (query.sum(-1) <= 0).any()
        ):
            raise ValueError("Trusted finite target neutral formula [B,10] is mandatory")
        if not torch.equal(
            (batch["template_formulas"] + batch["delta_formulas"])[mask],
            query[:, None].expand(-1, mask.shape[1], -1)[mask],
        ):
            raise ValueError("Signed formula delta must equal target minus template")
        if not mask.any(-1).all():
            raise ValueError("Zero-template target: report coverage and resolve explicitly")
        flat = self.structure_encoder(
            batch["graphs"] if self.encoder_kind == "graph" else batch["fingerprints"]
        )
        structure = flat.new_zeros((*mask.shape, 256))
        structure[mask] = flat
        tokens = self.projector(
            structure,
            batch["template_formulas"],
            batch["delta_formulas"],
            mask,
        )
        memory, global_state, padding, effective = self.template_set(tokens, mask)
        return {
            "chain_logits": self.chain_head(global_state),
            "template_tokens": tokens,
            "template_memory": memory,
            "template_global": global_state,
            "memory_padding_mask": padding,
            "effective_template_mask": effective,
        }

    def forward(self, batch: dict, polar_input_ids=None):
        output = self.encode_templates(batch)
        if polar_input_ids is None:
            polar_input_ids = batch["target_polar_token_ids"][:, :-1]
        output["polar_logits"] = self.polar_decoder(
            polar_input_ids, output["template_memory"], output["memory_padding_mask"]
        )
        return output

    @torch.no_grad()
    def generate(self, batch, beam_size=1, max_length=None):
        output = self.encode_templates(batch)
        output["polar_candidates"] = self.polar_decoder.generate(
            output["template_memory"], output["memory_padding_mask"], beam_size, max_length
        )
        return output
