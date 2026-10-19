"""Known chains determine polar formula; no learned chain output or chain loss."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool

from structure_pretrain.models.fingerprint_encoder import FingerprintEncoder
from structure_pretrain.models.graph_encoder import GraphEncoder
from structure_pretrain.models.polar_decoder import PolarDecoder
from structure_pretrain.models.template_set_transformer import TemplateSetTransformer
from structure_pretrain.models.template_token_projector import TemplateTokenProjector


class FragmentGraphEncoder(GraphEncoder):
    def __init__(self, config, dropout):
        super().__init__(config, dropout)
        self.marker_embedding = nn.Embedding(3, config.hidden_dim, padding_idx=0)

    def forward(self, graphs):
        state = self.atom_encoder(graphs.x) + self.marker_embedding(graphs.site_role)
        bonds = self.bond_encoder(graphs.edge_attr)
        for layer, norm in zip(self.layers, self.norms):
            state = self.dropout(norm(self.activation(layer(state, graphs.edge_index, bonds))))
        return self.readout(global_add_pool(state, graphs.batch, size=graphs.num_graphs))


class PolarOnlyModel(nn.Module):
    def __init__(self, config, tokenizer, statistics, chain_counts, scaffold_counts):
        super().__init__()
        if (config.structure_dim, config.formula_dim, config.d_model) != (256, 128, 512):
            raise ValueError("v0.4 requires 256D structure + 128D formula ->512D fusion")
        self.encoder_kind = config.encoder
        self.structure_encoder = (
            FragmentGraphEncoder(config.graph, config.dropout)
            if config.encoder == "graph"
            else FingerprintEncoder(config.dropout)
        )
        self.projector = TemplateTokenProjector(statistics, config.dropout)
        self.template_set = TemplateSetTransformer(config)
        self.polar_decoder = PolarDecoder(config, tokenizer)
        self.register_buffer(
            "known_chain_counts", torch.as_tensor(chain_counts, dtype=torch.float32)
        )
        self.register_buffer(
            "scaffold_counts", torch.as_tensor(scaffold_counts, dtype=torch.float32)
        )

    def polar_formula(self, full, chains):
        if (
            full.ndim != 2
            or full.shape[-1] != 10
            or chains.shape != (len(full), 2)
            or chains.dtype != torch.long
            or not torch.isfinite(full).all()
            or (full < 0).any()
            or not torch.equal(full, full.round())
            or (chains[:, 0] == 0).any()
            or (chains < 0).any()
            or (chains >= len(self.known_chain_counts)).any()
        ):
            raise ValueError("Trusted full formula and canonical known chain tokens required")
        result = full - self.scaffold_counts - self.known_chain_counts[chains].sum(1)
        result = result.clone()
        result[:, 1] -= ((chains != 0).sum(-1) == 1).to(result.dtype)
        if (result < 0).any() or (result[:, 4] < 1).any():
            raise ValueError("Impossible chain hypothesis / polar formula")
        return result

    def encode_templates(self, batch):
        mask = batch["template_mask"]
        polar = self.polar_formula(batch["target_full_formula"], batch["given_chain_tokens"])
        if not mask.any(-1).all() or not torch.equal(
            (batch["template_formulas"] + batch["delta_formulas"])[mask],
            polar[:, None].expand(-1, mask.shape[1], -1)[mask],
        ):
            raise ValueError(
                "Missing template or target polar delta differs from known-chain arithmetic"
            )
        embedding = self.structure_encoder(
            batch["graphs"] if self.encoder_kind == "graph" else batch["fingerprints"]
        )
        structure = embedding.new_zeros((*mask.shape, 256))
        structure[mask] = embedding
        tokens = self.projector(
            structure, batch["template_formulas"], batch["delta_formulas"], mask
        )
        memory, _, padding, effective = self.template_set(tokens, mask)
        return {
            "template_tokens": tokens,
            "template_memory": memory,
            "memory_padding_mask": padding,
            "effective_template_mask": effective,
            "target_polar_formula": polar,
        }

    def forward(self, batch):
        output = self.encode_templates(batch)
        output["polar_logits"] = self.polar_decoder(
            batch["target_polar_token_ids"][:, :-1],
            output["template_memory"],
            output["memory_padding_mask"],
        )
        return output

    @torch.no_grad()
    def generate(self, batch, beam_size=1, max_length=84):
        output = self.encode_templates(batch)
        output["polar_candidates"] = self.polar_decoder.generate(
            output["template_memory"], output["memory_padding_mask"], beam_size, max_length
        )
        return output


def polar_loss(prediction, batch, pad_id):
    return F.cross_entropy(
        prediction["polar_logits"].transpose(1, 2),
        batch["target_polar_token_ids"][:, 1:],
        ignore_index=pad_id,
    )


def candidate_polar_formulas(model, full_formulas, chain_candidates):
    """Future interface: keep N chain hypotheses independent; never pool their memories.

    Caller supplies a distinct template batch per hypothesis, flattens B,N into
    B*N for generate(), then restores candidate axes. No spectrum predictor exists here.
    Invalid hypotheses raise rather than being silently removed.
    """
    if chain_candidates.ndim != 3 or chain_candidates.shape[0] != len(full_formulas):
        raise ValueError("Expected known-chain candidates [B,N,2]")
    b, n, _ = chain_candidates.shape
    full = full_formulas[:, None].expand(-1, n, -1).reshape(b * n, 10)
    return model.polar_formula(full, chain_candidates.reshape(b * n, 2)).reshape(b, n, 10)


def independent_candidate_batch(
    model, full_formulas, chain_candidates, template_formulas, template_mask, structures
):
    """Prepare [B,N,K] references as B*N independent examples, never pool N.

    ``structures`` contains graphs or fingerprints in mask-true B,N,K order.
    Each chain hypothesis owns its references and its own derived polar formula.
    No target polar graph or spectral predictor is an argument.
    """
    polar = candidate_polar_formulas(model, full_formulas, chain_candidates)
    b, n, _ = polar.shape
    if (
        template_formulas.ndim != 4
        or template_formulas.shape[:2] != (b, n)
        or template_formulas.shape[-1] != 10
        or template_mask.shape != template_formulas.shape[:-1]
    ):
        raise ValueError("Independent templates must have shapes[B,N,K,10] and[B,N,K]")
    k = template_mask.shape[-1]
    return {
        "target_full_formula": full_formulas[:, None].expand(-1, n, -1).reshape(b * n, 10),
        "given_chain_tokens": chain_candidates.reshape(b * n, 2),
        "template_formulas": template_formulas.reshape(b * n, k, 10),
        "delta_formulas": (polar[:, :, None] - template_formulas).reshape(b * n, k, 10),
        "template_mask": template_mask.reshape(b * n, k),
        **structures,
    }
