import importlib.util
from pathlib import Path

import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "similarity_eval", "scripts/evaluate_polar_similarity_v04.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_prefix_before_target_maximum(monkeypatch):
    m = module(monkeypatch)
    assert m.topn_similarities([0.2, 0.1, 0.95, 0.7]) == {
        "1": 0.2,
        "3": 0.95,
        "5": 0.95,
        "10": 0.95,
    }
    assert m.topn_similarities([]) == dict.fromkeys(("1", "3", "5", "10"), 0.0)


def test_missing_candidate_keeps_all_target_denominator(monkeypatch):
    m = module(monkeypatch)
    rows = [
        {
            "candidate_count": 0,
            "s": dict.fromkeys(("1", "3", "5", "10"), 0.0),
            "s_all": 0.0,
            "exact": dict.fromkeys(("1", "3", "5", "10"), False),
            "template_best": 0.6,
            "template_top1": 0.5,
        },
        {
            "candidate_count": 1,
            "s": dict.fromkeys(("1", "3", "5", "10"), 0.8),
            "s_all": 0.8,
            "exact": dict.fromkeys(("1", "3", "5", "10"), False),
            "template_best": 0.4,
            "template_top1": 0.3,
        },
    ]
    result = m.sample_summary(rows)
    assert result["all_targets"]["1"]["similarity"]["mean"] == 0.4
    assert result["covered_targets"]["1"]["similarity"]["mean"] == 0.8
    assert result["all_targets"]["10"]["delta_vs_template_best"]["mean"] == pytest.approx(-0.1)
    assert result["all_targets"]["10"]["thresholds"]["0.8"]["fraction"] == 0.5


def test_delta_ties_and_fingerprint_one_are_not_graph_exact(monkeypatch):
    m = module(monkeypatch)
    gain = m.gain_summary([0.4, 0.5, 0.6], [0.4, 0.6, 0.5])
    assert [gain[k]["count"] for k in ("positive", "zero", "negative")] == [1, 1, 1]
    assert m.thresholds([1.0, 0.9, 0])["1.0"]["count"] == 1
    row = {
        "candidate_count": 1,
        "s": dict.fromkeys(("1", "3", "5", "10"), 1.0),
        "s_all": 1.0,
        "exact": dict.fromkeys(("1", "3", "5", "10"), False),
        "template_best": 0.8,
        "template_top1": 0.7,
    }
    assert m.sample_summary([row])["all_targets"]["1"]["tanimoto1_without_exact"]["count"] == 1


def test_frozen_restore_cache_never_cleanup_category(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "storage_audit", "scripts/audit_experiment_storage.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert loaded.category(loaded.WORKSPACE / "restored_v04/__pycache__/x.pyc").startswith(
        "protected"
    )
    assert "checkpoint_keep" in loaded.category(
        loaded.WORKSPACE / "phospholipid-structure-pretrain/outputs/x.pt"
    )
