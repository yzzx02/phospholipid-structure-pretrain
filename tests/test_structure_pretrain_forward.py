import torch

from structure_pretrain.config import ModelConfig
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.models.headgroup_decoder import HeadgroupDecoder
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel
from structure_pretrain.training.losses import pretraining_loss


def test_full_forward_and_chain_counts(model_inputs):
    config, dataset, stats, batch = model_inputs
    model = StructurePretrainModel(
        config.model, len(dataset.chain_vocab), dataset.tokenizer, stats
    ).eval()
    with torch.no_grad():
        output = model(batch)
    assert output["architecture_logits"].shape == (4, 2)
    assert output["chain_logits"].shape == (4, 4, 6)
    assert output["template_memory"].shape == (4, 11, 512)
    assert output["template_global"].shape == (4, 512)
    assert output["headgroup_logits"].shape == (
        4,
        batch["target_headgroup_token_ids"].shape[1] - 1,
        len(dataset.tokenizer),
    )
    assert batch["target_chain_tokens"].ne(0).sum(-1).tolist() == [1, 2, 3, 4]
    assert (batch["template_mask"].sum(-1) < 10).all()
    assert batch["delta_formulas"][~batch["formula_available"]].count_nonzero() == 0
    # There is deliberately no query formula token or input to the model.
    changed = dict(batch, target_formula=torch.full_like(batch["target_formula"], 99999))
    with torch.no_grad():
        second = model(changed)
    torch.testing.assert_close(output["template_global"], second["template_global"])
    losses = pretraining_loss(output, batch, dataset.tokenizer.pad_id, config.train)
    assert all(torch.isfinite(value) for value in losses.values())


def test_attachment_token_and_teacher_forcing_causal_mask():
    tokenizer = SmilesTokenizer()
    ids = tokenizer.pad(["[*:1]OCCN", "[*:1]O"])
    assert ids[0, 0] == tokenizer.bos_id
    assert ids[0, 1] == tokenizer.index["[*:1]"]
    assert tokenizer.decode(ids[0].tolist()) == "[*:1]OCCN"
    decoder = HeadgroupDecoder(ModelConfig(), tokenizer).eval()
    memory, padding = torch.randn(2, 4, 512), torch.zeros(2, 4, dtype=torch.bool)
    inputs = ids[:, :-1]
    changed = inputs.clone()
    changed[0, -1] = tokenizer.index["P"]
    with torch.no_grad():
        before, after = decoder(inputs, memory, padding), decoder(changed, memory, padding)
    torch.testing.assert_close(before[0, :-1], after[0, :-1], atol=1e-6, rtol=1e-5)
    assert ids.eq(tokenizer.eos_id).sum() == 2


def test_headgroup_pad_excluded_from_loss(model_inputs):
    config, dataset, stats, batch = model_inputs
    model = StructurePretrainModel(
        config.model, len(dataset.chain_vocab), dataset.tokenizer, stats
    ).eval()
    with torch.no_grad():
        output = model(batch)
    before = pretraining_loss(output, batch, 0, config.train)["headgroup_loss"]
    altered = dict(output)
    altered["headgroup_logits"] = output["headgroup_logits"].clone()
    altered["headgroup_logits"][batch["target_headgroup_token_ids"][:, 1:].eq(0)] = 99999
    after = pretraining_loss(altered, batch, 0, config.train)["headgroup_loss"]
    torch.testing.assert_close(before, after)
