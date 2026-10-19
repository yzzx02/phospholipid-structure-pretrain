"""Process-local attention selection for controlled ROCm diagnosis."""

from contextlib import contextmanager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def attention_backend(name):
    if name == "default":
        yield
        return
    if name != "math":
        raise ValueError(f"Unknown attention backend: {name}")
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
