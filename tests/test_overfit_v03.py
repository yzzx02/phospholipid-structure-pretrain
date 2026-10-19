import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.overfit_pretraining_v03 import THRESHOLDS, gate_passed, select_rows
from structure_pretrain.data.real_structure_dataset import sample_targets


@pytest.fixture()
def sample_metadata(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    group_splits = np.array([0] * 34 + [1] * 3 + [2] * 3)
    np.save(prepared / "group_splits.npy", group_splits)
    np.save(prepared / "polar_groups.npy", np.repeat(np.arange(40), 2))
    np.save(prepared / "row_splits.npy", np.repeat(group_splits, 2))
    np.save(prepared / "chain_labels.npy", np.ones((80, 2), dtype=np.int64))
    (prepared / "chain_vocab.json").write_text(json.dumps({"tokens": ["NULL", "A-16:0"]}))
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "token_length": 12 + group if group < 34 else 200,
                    "chemistry_family": "phosphate",
                    "polar_plausibility_tier": "CORE",
                    "structure_origin": "fixture",
                }
                for group in range(40)
            ]
        ),
        prepared / "polar_groups.parquet",
    )
    return prepared


def test_overfit_selection_keeps_train_heads_and_train_length_extremes(sample_metadata):
    rows = select_rows(sample_metadata, 32, 42)
    groups = np.load(sample_metadata / "polar_groups.npy")[rows]
    np.testing.assert_array_equal(rows, select_rows(sample_metadata, 32, 42))
    assert len(np.unique(groups)) == 32
    assert {0, 33}.issubset(groups.tolist())
    assert np.all(np.load(sample_metadata / "row_splits.npy")[rows] == 0)


def test_overfit_gate_requires_eos_validity_and_target_reconstruction():
    assert gate_passed(dict(THRESHOLDS))
    for key in THRESHOLDS:
        metrics = dict.fromkeys(THRESHOLDS, 1.0)
        metrics[key] = 0.0
        assert not gate_passed(metrics)


def test_smoke_sampling_preserves_frozen_delivered_files(sample_metadata, tmp_path):
    for split in ("train", "validation"):
        (sample_metadata / f"local_smoke_{split}_rows.npy").write_bytes(b"frozen transfer asset")
    before = {p.name: p.read_bytes() for p in sample_metadata.iterdir()}
    output = tmp_path / "run" / "sample_rows"
    samples = sample_targets(sample_metadata, 8, 4, sample_output=output)
    assert before == {p.name: p.read_bytes() for p in sample_metadata.iterdir()}
    for split, count in (("train", 8), ("validation", 4)):
        assert len(samples[split]) == count
        np.testing.assert_array_equal(
            samples[split], np.load(output / f"local_smoke_{split}_rows.npy")
        )
