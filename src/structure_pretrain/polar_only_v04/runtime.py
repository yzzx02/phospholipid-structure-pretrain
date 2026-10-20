"""Shared reproducible construction for v0.4 only."""

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.polar_only_v04.config import PolarConfig, validate_config
from structure_pretrain.polar_only_v04.model import PolarOnlyModel
from structure_pretrain.polar_only_v04.retrieval import statistics_path


def load_config(path, kind="smoke"):
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(PolarConfig()), OmegaConf.load(path))
    )
    if kind == "overfit":
        config.model.dropout = 0.0
        config.model.template_dropout = 0.0
        config.train.learning_rate = 0.0003
        config.train.batch_size = 8
        config.train.gradient_accumulation = 1
        config.train.num_workers = 0
        config.train.checkpoint_every = 200
    return validate_config(config)


def make_model(config, tokenizer, root):
    root = Path(root)
    stats = read_json(statistics_path(root, config.max_top_k, config.h_fraction, config.train.seed))
    if not stats["training_pairs_only"] or (stats["K"], stats["h_fraction"], stats["seed"]) != (
        config.max_top_k,
        config.h_fraction,
        config.train.seed,
    ):
        raise ValueError("Untrusted standardizer")
    return PolarOnlyModel(
        config.model,
        tokenizer,
        stats,
        np.load(root / "chain_counts.npy"),
        np.load(root / "scaffold_counts.npy"),
    )


def move(batch, device):
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def require_gpu(root, backend="rocm"):
    if (Path(root) / "GPU_STOP.json").exists():
        raise RuntimeError("GPU watchdog previously stopped v0.4; no automatic retry")
    if backend == "native_cuda":
        if sys.platform != "win32" or torch.version.hip or not torch.version.cuda:
            raise RuntimeError("Native Windows NVIDIA CUDA build required")
        if not torch.cuda.is_available() or "RTX 4070" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This bounded native CUDA experiment requires an RTX 4070 GPU")
        return torch.device("cuda")
    if backend != "rocm":
        raise ValueError(f"Unknown GPU backend: {backend}")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Verified HIP GPU required; no silent CPU fallback")
    if "9070 XT" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Unexpected device; this config is bounded to verified RX9070XT")
    return torch.device("cuda")
