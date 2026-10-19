import pytest
import torch
from torch import nn

from structure_pretrain.training.checkpoint_validation import validate_reload


class Prediction(nn.Module):
    def __init__(self, offset=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.offset = offset

    def forward(self, batch):
        value = self.weight + self.offset
        return {"chain_logits": value, "polar_logits": value}


def test_reload_rejects_changed_weights_even_below_logit_tolerance():
    original, restored = Prediction(), Prediction()
    with torch.no_grad():
        restored.weight.add_(1e-8)
    with pytest.raises(AssertionError):
        validate_reload(original, restored, {}, allow_roundoff=True)


def test_reload_reports_small_roundoff_but_cpu_stays_exact():
    original, restored = Prediction(), Prediction(2e-6)
    _, report = validate_reload(original, restored, {}, allow_roundoff=True)
    assert report["checkpoint_state_dict_identical"]
    assert not report["checkpoint_reload_identical_logits"]
    assert report["checkpoint_reload_max_abs_logit_difference"]["polar_logits"] > 0
    with pytest.raises(AssertionError):
        validate_reload(original, restored, {})


def test_reload_rejects_output_changes_beyond_float32_defaults():
    with pytest.raises(AssertionError):
        validate_reload(Prediction(), Prediction(1e-3), {}, allow_roundoff=True)
