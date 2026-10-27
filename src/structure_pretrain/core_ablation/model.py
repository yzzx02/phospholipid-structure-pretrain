"""Capacity-matched formula-only model: no retrieval inputs or structural features."""

import torch

from structure_pretrain.core_templates.model import GlycerolCoreModel


class FormulaOnlyModel(GlycerolCoreModel):
    def __init__(self, config, tokenizer, statistics):
        super().__init__(config, tokenizer, statistics)
        self.register_buffer("formula_input_indices", torch.arange(2048) % 10)

    def encode_templates(self, batch):
        if set(batch) != {"target_core_formula"}:
            raise ValueError("Formula-only condition accepts ONLY the target formula")
        formula = batch["target_core_formula"]
        if formula.ndim != 2 or formula.shape[1] != 10 or not torch.isfinite(formula).all():
            raise ValueError("Target core formula[B,10] required")
        if (formula < 0).any() or not torch.equal(formula, formula.round()):
            raise ValueError("Nonnegative integral formula required")
        normalizer = self.projector.normalizer
        z = (formula - normalizer.template_mean) / normalizer.template_std
        # Fixed expansion, not a fingerprint. All original encoder weights remain
        # active/trainable, with exactly the same parameter shapes and total count.
        features = z[:, self.formula_input_indices]
        features = features[:, None].expand(-1, 3, -1).reshape(-1, 2048)
        structure = self.structure_encoder(features).reshape(len(formula), 3, 256)
        inputs = formula[:, None].expand(-1, 3, -1)
        mask = torch.ones((len(formula), 3), dtype=torch.bool, device=formula.device)
        tokens = self.projector(structure, inputs, torch.zeros_like(inputs), mask)
        memory, _, padding, effective = self.template_set(tokens, mask)
        return {
            "template_memory": memory,
            "memory_padding_mask": padding,
            "effective_template_mask": effective,
            "target_core_formula": formula,
        }

    def forward(self, batch):
        if set(batch) != {"target_core_formula", "target_polar_token_ids"}:
            raise ValueError("Formula-only training input must not contain reference data")
        output = self.encode_templates({"target_core_formula": batch["target_core_formula"]})
        output["polar_logits"] = self.polar_decoder(
            batch["target_polar_token_ids"][:, :-1],
            output["template_memory"],
            output["memory_padding_mask"],
        )
        return output
