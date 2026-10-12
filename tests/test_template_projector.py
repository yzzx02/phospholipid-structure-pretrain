import pytest
import torch

from structure_pretrain.models.template_token_projector import TemplateTokenProjector
from structure_pretrain.utils.formula import FormulaStandardizer, validate_formula


def test_direct_278_to_512_and_missing_formula():
    stats = {"delta_mean": [5.0] * 10, "delta_std": [2.0] * 10}
    projector = TemplateTokenProjector(stats).eval()
    structure = torch.randn(2, 3, 256)
    formula, delta = torch.ones(2, 3, 10), torch.randn(2, 3, 10)
    available = torch.tensor([[True, False, False], [False, True, False]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    similarity = torch.rand(2, 3)
    changed = delta.clone()
    changed[~available] = 1e8
    first = projector(structure, formula, delta, similarity, available, mask)
    second = projector(structure, formula, changed, similarity, available, mask)
    assert first.shape == (2, 3, 512)
    torch.testing.assert_close(first, second)
    assert first[~mask].count_nonzero() == 0
    template_z, delta_z = projector.normalizer(formula, delta, available)
    assert delta_z[~available].count_nonzero() == 0
    assert template_z[~available].count_nonzero() > 0
    assert projector.mlp[0].in_features == 278


def test_training_statistics_exclude_padding_and_missing_delta():
    item = {
        "template_mask": torch.tensor([True, True, False]),
        "formula_available": torch.tensor([True, False, False]),
        "template_formulas": torch.tensor([[1.0] * 10, [3.0] * 10, [999.0] * 10]),
        "delta_formulas": torch.tensor([[-2.0] * 10, [999.0] * 10, [999.0] * 10]),
    }
    stats = FormulaStandardizer.fit([item])
    assert stats["template_mean"] == [2.0] * 10
    assert stats["template_std"] == [1.0] * 10
    assert stats["delta_mean"] == [-2.0] * 10
    assert stats["delta_std"] == [1.0] * 10


@pytest.mark.parametrize("formula", [[1] * 9, [-1] + [0] * 9, [1.5] * 10])
def test_invalid_supplied_formula(formula):
    with pytest.raises(ValueError):
        validate_formula(formula)
