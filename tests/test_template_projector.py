import pytest
import torch

from structure_pretrain.models.template_token_projector import TemplateTokenProjector
from structure_pretrain.utils.formula import FormulaStandardizer, validate_formula


def test_independent_formula_embedding_and_fusion():
    stats = {"delta_mean": [5.0] * 10, "delta_std": [2.0] * 10}
    projector = TemplateTokenProjector(stats).eval()
    structure = torch.randn(2, 3, 256)
    formula, delta = torch.ones(2, 3, 10), torch.randn(2, 3, 10)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    first = projector(structure, formula, delta, mask)
    assert first.shape == (2, 3, 512)
    assert first[~mask].count_nonzero() == 0
    template_z, delta_z = projector.normalizer(formula, delta)
    torch.testing.assert_close(delta_z, (delta - 5) / 2)
    assert projector.formula_mlp[0].in_features == 20
    assert projector.formula_mlp[0].out_features == 128
    assert projector.fusion_mlp[0].in_features == 384
    with pytest.raises(ValueError, match="mandatory"):
        projector(structure, formula * float("nan"), delta, mask)
    with pytest.raises(ValueError, match="Missing"):
        projector(structure, torch.zeros_like(formula), delta, mask)


def test_streaming_training_statistics_exclude_padding():
    item = {
        "template_mask": torch.tensor([True, True, False]),
        "template_formulas": torch.tensor([[1.0] * 10, [3.0] * 10, [999.0] * 10]),
        "delta_formulas": torch.tensor([[-2.0] * 10, [2.0] * 10, [999.0] * 10]),
    }
    stats = FormulaStandardizer.fit([item])
    assert stats["template_mean"] == [2.0] * 10
    assert stats["template_std"] == [1.0] * 10
    assert stats["delta_mean"] == [0.0] * 10
    assert stats["delta_std"] == [2.0] * 10
    assert stats["pair_count"] == 2


@pytest.mark.parametrize("formula", [[1] * 9, [-1] + [0] * 9, [1.5] * 10])
def test_invalid_supplied_formula(formula):
    with pytest.raises(ValueError):
        validate_formula(formula)
