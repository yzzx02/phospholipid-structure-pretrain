"""Host migration must preserve device guards, fault latches and checkpoint protection."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from structure_pretrain.polar_only_v04 import runtime
from structure_pretrain.utils.resources import peak_process_memory_kib

spec = importlib.util.spec_from_file_location(
    "native_supervisor",
    Path(__file__).resolve().parents[1] / "scripts/supervise_formula_beam_cuda_v04.py",
)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def test_native_cuda_checks_do_not_weaken_rocm_or_stop_latch(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.torch.version, "hip", None)
    monkeypatch.setattr(runtime.torch.version, "cuda", "12.8")
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "get_device_name", lambda index: "RTX 4070 SUPER")
    assert str(runtime.require_gpu(tmp_path, "native_cuda")) == "cuda"
    with pytest.raises(RuntimeError, match="HIP"):
        runtime.require_gpu(tmp_path)
    (tmp_path / "GPU_STOP.json").write_text("{}")
    with pytest.raises(RuntimeError, match="no automatic retry"):
        runtime.require_gpu(tmp_path, "native_cuda")


def test_native_cuda_never_falls_back_to_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.torch.version, "hip", None)
    monkeypatch.setattr(runtime.torch.version, "cuda", "12.8")
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="RTX 4070"):
        runtime.require_gpu(tmp_path, "native_cuda")


def test_host_monitor_detects_changed_dump_and_access_loss():
    before = {"files": {"old.dmp": [100, 1000]}, "unreadable_directories": []}
    after = {
        "files": {"old.dmp": [101, 1000], "new.dmp": [102, 2000]},
        "unreadable_directories": [],
    }
    assert supervisor.changed_dumps(before, after) == ["old.dmp", "new.dmp"]
    after["unreadable_directories"] = ["WATCHDOG"]
    with pytest.raises(RuntimeError, match="access changed"):
        supervisor.changed_dumps(before, after)


def test_peak_memory_is_available_on_this_platform():
    assert peak_process_memory_kib() > 0


def test_native_report_rejects_overlapping_slices(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "native_report",
        Path(__file__).resolve().parents[1] / "scripts/report_formula_beam_cuda_v04.py",
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    contract = {"encoder": "fingerprint", "mode": "raw", "beam_size": 5, "start": 0, "count": 1}
    host = {
        **contract,
        "gpu_backend": "native_cuda",
        "child_exit_code": 0,
        "stop_reason": None,
        "new_windows_gpu_dump_count": 0,
        "display_reset_count": 0,
        "checkpoints_unchanged": True,
        "training_started": False,
    }
    metric = {
        **contract,
        "status": "completed",
        "cohort_rows_sha256": report.EXPECTED_COHORT,
        "gpu_backend": "native_cuda",
        "hip": None,
        "precision": "fp32",
        "attention_backend": "math",
        "training_started": False,
        "inputs": {},
        "elapsed_seconds": 0.1,
        "peak_RAM_kib": 1,
        "peak_VRAM_allocated_bytes": 1,
        "torch": "2.9.1+cu128",
        "cuda": "12.8",
        "gpu_name": "RTX 4070 SUPER",
    }
    for name in ("job_a", "job_b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "host_monitor.json").write_text(json.dumps(host))
        (folder / "metrics.json").write_text(json.dumps(metric))
        (folder / "predictions_private.jsonl").write_text(
            json.dumps({"cohort_index": 0, "row": 10}) + "\n"
        )
    with pytest.raises(ValueError, match="Overlapping"):
        report.aggregate_jobs(tmp_path, np.arange(10, 522))
