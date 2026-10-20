"""Incomplete or substituted target cohorts must never become a full result."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "formula_report", Path(__file__).resolve().parents[1] / "scripts/report_formula_beam_v04.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.mark.parametrize(
    "records",
    [
        [{"cohort_index": 0, "row": 10}],
        [{"cohort_index": 0, "row": 10}, {"cohort_index": 0, "row": 10}],
        [{"cohort_index": 0, "row": 10}, {"cohort_index": 1, "row": 99}],
    ],
)
def test_report_rejects_missing_duplicate_and_substituted_targets(records):
    with pytest.raises(ValueError):
        report.check_records(records, np.array([10, 11]), 0, 2)


def test_failure_stages_keep_targets_with_no_survivor_and_formula_errors():
    records = [
        {"raw_candidates": [], "accepted_unique": []},
        {"raw_candidates": [{"eos": False}], "accepted_unique": []},
        {
            "raw_candidates": [
                {"eos": True, "polar_valid": True, "polar_formula_match_including_H": False}
            ],
            "accepted_unique": [],
        },
    ]
    assert report.failure_stages(records) == {
        "search_has_no_surviving_beam": 1,
        "no_EOS": 1,
        "no_exact_Polar_formula_including_H": 1,
    }


def test_watchdog_job_partial_targets_never_enter_main_comparison(tmp_path):
    folder = tmp_path / "fingerprint_elements_b20_s0"
    folder.mkdir()
    host = {
        "child_exit_code": 15,
        "stop_reason": "new_windows_gpu_fault_dump",
        "new_windows_gpu_dump_count": 1,
        "display_reset_count": 0,
        "training_started": False,
    }
    (folder / "host_monitor.json").write_text(json.dumps(host))
    partial = [
        {
            "cohort_index": i,
            "row": 100 + i,
            "raw_candidates": [],
            "accepted_unique": [],
            "decode_seconds": 0.1,
            "validation_seconds": 0.1,
        }
        for i in range(2)
    ]
    (folder / "predictions_private.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in partial)
    )
    rows = np.arange(100, 612)
    with pytest.raises(ValueError, match="Unsafe or failed"):
        report.aggregate_case(tmp_path, "fingerprint", "elements", 20, rows, False)
    case = report.aggregate_case(tmp_path, "fingerprint", "elements", 20, rows, True)
    assert case["status"] == "stopped"
    assert case["evaluated_targets"] == 0 and "raw" not in case
    assert case["failed_jobs"][0]["partial_completed_targets"] == 2
    assert not case["failed_jobs"][0]["included_in_main_comparison"]
