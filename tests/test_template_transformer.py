import pytest
import torch

from structure_pretrain.config import ModelConfig
from structure_pretrain.models.template_set_transformer import TemplateSetTransformer
from structure_pretrain.utils.masks import drop_templates


def test_permutation_and_padding_invariance():
    model = TemplateSetTransformer(ModelConfig()).eval()
    tokens = torch.randn(2, 10, 512)
    mask = torch.tensor([[True] * 3 + [False] * 7, [True] + [False] * 9])
    permutation = torch.tensor([2, 0, 1, 9, 7, 6, 5, 8, 4, 3])
    with torch.no_grad():
        memory, global_state, padding, _ = model(tokens, mask)
        _, reordered, _, _ = model(tokens[:, permutation], mask[:, permutation])
        corrupted = tokens.clone()
        corrupted[~mask] = 1e8
        _, unaffected, _, _ = model(corrupted, mask)
    assert memory.shape == (2, 11, 512)
    torch.testing.assert_close(global_state, reordered, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(global_state, unaffected)
    assert memory[padding].count_nonzero() == 0


@pytest.mark.parametrize("probability", [0.0, 0.2, 1.0])
def test_dropout_retains_one_and_updates_mask(probability):
    torch.manual_seed(4)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    result = drop_templates(mask, probability, True)
    assert result.any(-1).all()
    assert not result[~mask].any()
    if probability == 1:
        assert result.sum(-1).tolist() == [1, 1]
    model = TemplateSetTransformer(ModelConfig(template_dropout=probability)).train()
    _, _, padding, effective = model(torch.randn(2, 3, 512), mask)
    assert torch.equal(padding[:, 1:], ~effective)
    assert not padding[:, 0].any()
    assert torch.equal(drop_templates(mask, probability, False), mask)


def test_reject_zero_templates():
    with pytest.raises(ValueError):
        drop_templates(torch.zeros(2, 10, dtype=torch.bool), 0.2, True)
