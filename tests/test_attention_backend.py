import pytest
import torch

from structure_pretrain.utils.attention import attention_backend


def flags():
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.mha.get_fastpath_enabled(),
    )


def test_math_attention_restores_backend_flags_after_exception():
    before = flags()
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        with attention_backend("math"):
            assert flags() == (False, False, True, False)
            raise RuntimeError("diagnostic failure")
    assert flags() == before


def test_math_attention_preserves_complete_model_outputs(model_inputs):
    from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel

    config, dataset, statistics, batch = model_inputs
    model = StructurePretrainModel(
        config.model, len(dataset.chain_vocab), dataset.tokenizer, statistics
    ).eval()
    with torch.no_grad():
        reference = model(batch)
        with attention_backend("math"):
            actual = model(batch)
    for name in ("chain_logits", "polar_logits"):
        torch.testing.assert_close(actual[name], reference[name], rtol=1e-4, atol=1e-5)
