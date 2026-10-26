"""FP32/Math native-CUDA runtime and complete random-state restart helpers."""

import random

import numpy as np
import torch

from structure_pretrain.core_templates.common import configuration


def initialize(seed=42):
    torch.set_num_threads(configuration()["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state(device):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"].cpu())
    if str(device).startswith("cuda"):
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def cpu_copy(value):
    import copy

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def compare(a, b, tolerance):
    if isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
        return 0.0
    if torch.is_tensor(a):
        assert a.shape == b.shape and a.dtype == b.dtype
        maximum = float((a - b).abs().max()) if a.numel() else 0.0
        if maximum > tolerance:
            raise AssertionError(f"Full-state restart differs: {maximum} > {tolerance}")
        return maximum
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((compare(a[k], b[k], tolerance) for k in a), default=0.0)
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        return max((compare(x, y, tolerance) for x, y in zip(a, b)), default=0.0)
    assert a == b
    return 0.0
