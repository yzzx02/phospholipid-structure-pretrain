"""Exact checkpoint state validation with dtype-aware GPU output comparison."""

import torch


@torch.no_grad()
def validate_reload(model, restored, batch, allow_roundoff=False):
    # Serialization must preserve every parameter and buffer exactly on every device.
    torch.testing.assert_close(model.state_dict(), restored.state_dict(), rtol=0, atol=0)
    reference, reloaded = model(batch), restored(batch)
    identical, differences = True, {}
    for name in ("chain_logits", "polar_logits"):
        identical &= torch.equal(reference[name], reloaded[name])
        differences[name] = float((reference[name] - reloaded[name]).abs().max())
        # GPU graph reductions need not be bitwise reproducible. PyTorch's documented
        # float32 defaults are rtol=1.3e-6 / atol=1e-5; CPU retains exact comparison.
        tolerance = {} if allow_roundoff else {"rtol": 0, "atol": 0}
        torch.testing.assert_close(reference[name], reloaded[name], **tolerance)
    return reference, {
        "checkpoint_state_dict_identical": True,
        "checkpoint_reload_identical_logits": identical,
        "checkpoint_reload_logits_close": True,
        "checkpoint_reload_max_abs_logit_difference": differences,
        "checkpoint_reload_comparison": "pytorch_dtype_defaults" if allow_roundoff else "exact",
    }
