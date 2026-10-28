"""Exact frozen template batches with an explicit ban on test-target access."""

import numpy as np
import torch

from structure_pretrain.core_lr_study.common import BASE_ROOT, ROOT, parameter_digest
from structure_pretrain.core_templates.data import CoreData as OriginalData
from structure_pretrain.core_templates.data import make_model as original_model
from structure_pretrain.core_templates.runtime import initialize, restore_rng


class StudyData(OriginalData):
    def __init__(self, root=ROOT, device="cpu"):
        super().__init__(BASE_ROOT, device)
        self.study_root = root

    def indices(self, split):
        if split not in (0, 1):
            raise ValueError("Test split forbidden in LR schedule selection")
        return super().indices(split)

    def batch(self, indices, k=3):
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(self.splits[indices] == 2):
            raise ValueError("Test labels/conditions forbidden in LR study")
        return super().batch(indices, k)


def make_model(root=ROOT):
    return original_model(BASE_ROOT)


def load_shared_initialization(model, root=ROOT, device="cpu"):
    initialize(42)
    saved = torch.load(
        root / "data/shared_initial_state.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(saved["model"])
    assert parameter_digest(model) == saved["parameter_sha256"]
    # Initial snapshot is CPU-only. CUDA dropout starts at the exact seed42 state,
    # as in the original fresh E3 run; no old model weights warm-start new runs.
    restore_rng(saved["rng"], "cpu")
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(42)
    return saved["parameter_sha256"]
