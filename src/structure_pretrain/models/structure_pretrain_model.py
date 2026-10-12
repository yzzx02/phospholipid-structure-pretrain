from torch import nn

from structure_pretrain.config import ModelConfig
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer

from .architecture_head import ArchitectureHead
from .chain_head import ChainHead
from .fingerprint_encoder import FingerprintEncoder
from .graph_encoder import GraphEncoder
from .headgroup_decoder import HeadgroupDecoder
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
        self.encoder_kind = config.encoder
        if config.encoder == "graph":
            self.structure_encoder = GraphEncoder(config.graph, config.dropout)
        elif config.encoder == "fingerprint":
            self.structure_encoder = FingerprintEncoder(config.dropout)
        else:
            raise ValueError("Unknown encoder")
        self.projector = TemplateTokenProjector(formula_statistics, config.dropout)
        self.template_set = TemplateSetTransformer(config)
        self.architecture_head = ArchitectureHead(config.d_model)
        self.chain_head = ChainHead(chain_vocab_size, config.d_model)
        self.headgroup_decoder = HeadgroupDecoder(config, tokenizer)

    def forward(self, batch: dict, headgroup_input_ids=None):
        mask = batch["template_mask"]
        if not mask.any(-1).all():
            raise ValueError("Skip targets with no retrievable templates")
        flat = self.structure_encoder(
            batch["graphs"] if self.encoder_kind == "graph" else batch["fingerprints"]
        )
        structure = flat.new_zeros((*mask.shape, 256))
        structure[mask] = flat
        tokens = self.projector(
            structure,
            batch["template_formulas"],
            batch["delta_formulas"],
            batch["structure_similarity"],
            batch["formula_available"],
            mask,
        )
        memory, global_state, padding, effective = self.template_set(tokens, mask)
        if headgroup_input_ids is None:
            headgroup_input_ids = batch["target_headgroup_token_ids"][:, :-1]
        return {
            "architecture_logits": self.architecture_head(global_state),
            "chain_logits": self.chain_head(global_state),
            "headgroup_logits": self.headgroup_decoder(headgroup_input_ids, memory, padding),
            "template_tokens": tokens,
            "template_memory": memory,
            "template_global": global_state,
            "memory_padding_mask": padding,
            "effective_template_mask": effective,
        }
