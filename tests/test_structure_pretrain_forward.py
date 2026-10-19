import pytest
import torch

from structure_pretrain.config import ModelConfig
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.models.polar_decoder import PolarDecoder
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel
from structure_pretrain.training.losses import pretraining_loss


def test_full_forward_and_chain_counts(model_inputs):
    config, dataset, stats, batch = model_inputs
    model = StructurePretrainModel(
        config.model, len(dataset.chain_vocab), dataset.tokenizer, stats
    ).eval()
    with torch.no_grad():
        output = model(batch)
    assert set(name for name in output if name.endswith("logits")) == {
        "chain_logits",
        "polar_logits",
    }
    assert output["chain_logits"].shape == (4, 2, 6)
    assert output["template_memory"].shape == (4, 11, 512)
    assert output["template_global"].shape == (4, 512)
    assert output["polar_logits"].shape == (
        4,
        batch["target_polar_token_ids"].shape[1] - 1,
        len(dataset.tokenizer),
    )
    assert batch["target_chain_tokens"].ne(0).sum(-1).tolist() == [1, 2, 2, 2]
    assert (batch["template_mask"].sum(-1) < 10).all()
    assert "formula_available" not in batch and "structure_similarity" not in batch
    # Target formula is mandatory and must agree with the signed per-template deltas.
    changed = dict(batch, target_formula=torch.full_like(batch["target_formula"], 99999))
    with pytest.raises(ValueError, match="Signed formula delta"):
        model(changed)
    losses = pretraining_loss(output, batch, dataset.tokenizer.pad_id, config.train)
    assert all(torch.isfinite(value) for value in losses.values())
    inference_batch = {k: v for k, v in batch.items() if k != "target_polar_token_ids"}
    free = model.generate(inference_batch, max_length=4)
    assert not free["chain_logits"].requires_grad
    assert len(free["polar_candidates"]) == 4


def test_attachment_token_and_teacher_forcing_causal_mask():
    tokenizer = SmilesTokenizer()
    ids = tokenizer.pad(["[*:1]P(=O)(O)OCCN", "[*:1]P(=O)(O)O"])
    assert ids[0, 0] == tokenizer.bos_id
    assert ids[0, 1] == tokenizer.index["[*:1]"]
    assert tokenizer.decode(ids[0].tolist()) == "[*:1]P(=O)(O)OCCN"
    decoder = PolarDecoder(ModelConfig(), tokenizer).eval()
    memory, padding = torch.randn(2, 4, 512), torch.zeros(2, 4, dtype=torch.bool)
    inputs = ids[:, :-1]
    changed = inputs.clone()
    changed[0, -1] = tokenizer.index["P"]
    with torch.no_grad():
        before, after = decoder(inputs, memory, padding), decoder(changed, memory, padding)
    torch.testing.assert_close(before[0, :-1], after[0, :-1], atol=1e-6, rtol=1e-5)
    assert ids.eq(tokenizer.eos_id).sum() == 2


def test_polar_pad_excluded_from_loss(model_inputs):
    config, dataset, stats, batch = model_inputs
    model = StructurePretrainModel(
        config.model, len(dataset.chain_vocab), dataset.tokenizer, stats
    ).eval()
    with torch.no_grad():
        output = model(batch)
    before = pretraining_loss(output, batch, 0, config.train)["polar_loss"]
    altered = dict(output)
    altered["polar_logits"] = output["polar_logits"].clone()
    altered["polar_logits"][batch["target_polar_token_ids"][:, 1:].eq(0)] = 99999
    after = pretraining_loss(altered, batch, 0, config.train)["polar_loss"]
    torch.testing.assert_close(before, after)
