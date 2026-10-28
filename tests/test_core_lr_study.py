"""Numerical schedule/selection invariants and frozen validation-only conditions."""

import numpy as np
import pytest
import torch

from structure_pretrain.core_lr_study.common import ROOT, configuration, parameter_digest, read
from structure_pretrain.core_lr_study.data import StudyData, load_shared_initialization, make_model
from structure_pretrain.core_lr_study.schedule import Schedule, stop_for_patience
from structure_pretrain.formal_v2.schedule import WarmupCosine
from structure_pretrain.reported_headgroups.formal_template_data import update_selection


def schedule(kind, warm=0):
    return Schedule(kind, 3e-4, 1e-5, warm, 427200)


@pytest.mark.parametrize("u", [1, 5340, 10000, 267000, 427200])
def test_constant(u):
    assert schedule("constant").value(u) == 3e-4


def test_cosine_endpoints_and_cycle():
    s = schedule("cosine")
    assert s.value(1) == 3e-4
    assert s.value(427200) == 1e-5
    assert s.value(267000) > 1e-5
    assert s.value(5340) > s.value(256320) > s.value(427200)


@pytest.mark.parametrize("u", [1, 3999, 4000, 4001, 256320, 256321, 267000, 427200])
def test_E3_matches_frozen_schedule(u):
    old = WarmupCosine(3e-4, 1e-5, 4000, 427200)
    assert schedule("warmup_cosine", 4000).value(u) == old.value(u)


def test_E4_warmup_10000():
    s = schedule("warmup_cosine", 10000)
    assert s.value(1) == 3e-8
    assert s.value(10000) == 3e-4
    assert s.value(10001) < 3e-4
    assert s.value(427200) == 1e-5


@pytest.mark.parametrize("kind,warm", [("constant", 0), ("cosine", 0), ("warmup_cosine", 10000)])
def test_restart_clock(kind, warm):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    s = schedule(kind, warm)
    for u in range(3):
        assert s.before_optimizer_step(opt, u) == s.value(u + 1)
        with pytest.raises(ValueError):
            s.state_dict()
        s.after_optimizer_step()
    other = schedule(kind, warm)
    other.load_state_dict(s.state_dict())
    assert other.before_optimizer_step(opt, 3) == s.value(4)
    with pytest.raises(ValueError):
        other.before_optimizer_step(opt, 3)


def test_import_old_E3_clock_without_rewarmup():
    s = schedule("warmup_cosine", 4000)
    old = {"specification": s.specification, "updates_completed": 256320}
    s.load_state_dict(old)
    assert s.updates_completed == 256320
    assert 1e-4 < s.value(256321) < 1.2e-4


@pytest.mark.parametrize(
    "epoch,bad,expected",
    [(43, 5, False), (48, 5, False), (49, 6, False), (50, 7, True), (50, 4, False), (57, 5, True)],
)
def test_patience_deferred_until_common_plot_window(epoch, bad, expected):
    assert stop_for_patience(epoch, bad) == expected


def test_global_best_and_counter_not_reset_at_epoch50():
    best, bad, improved = update_selection(0.095, 6, 0.096)
    assert best == 0.095 and bad == 7 and not improved
    assert stop_for_patience(50, bad)
    best, bad, improved = update_selection(best, bad, 0.094)
    assert best == 0.094 and bad == 0 and improved
    assert not stop_for_patience(50, bad)


def test_data_and_all_new_models_same_start():
    data = StudyData(ROOT, "cpu")
    assert len(data.indices(0)) == 170872 and len(data.indices(1)) == 9493
    with pytest.raises(ValueError, match="Test"):
        data.indices(2)
    with pytest.raises(ValueError, match="Test"):
        data.batch(np.flatnonzero(data.splits == 2)[:1])
    hashes = []
    for _ in ("E1", "E2", "E4"):
        model = make_model()
        hashes.append(load_shared_initialization(model))
        assert parameter_digest(model) == hashes[-1]
    assert len(set(hashes)) == 1
    assert hashes[0] == read(ROOT / "data/manifest.json")["initial_parameter_sha256"]


def test_frozen_design():
    c = configuration()
    assert c["minimum_epochs"] == 50 and c["max_epochs"] == 80
    assert c["planned_schedule_epochs"] == 80
    assert c["generation"]["split"] == "validation"
    assert c["generation"]["targets"] == 9493 and c["generation"]["beam_size"] == 32
    assert not c["patience_counter_resets_at_minimum_epoch"]
    assert not c["test_used_for_training_or_model_selection"]
