import json

import pyarrow.parquet as pq

from structure_pretrain.library_builder.pipeline import build_library
from structure_pretrain.library_builder.schemas import load_build_config

from .conftest import EXAMPLES


def test_report_and_all_required_artifacts(inputs, tmp_path):
    _, _, config = inputs
    output = tmp_path / "report"
    report = build_library(
        config,
        EXAMPLES / "chains.tsv",
        EXAMPLES / "polar_moieties.tsv",
        output,
        EXAMPLES / "p_to_o_mapping.tsv",
    )
    assert report["total_structures"] == report["unique_canonical_smiles"] == 60
    assert (
        report["AA_count"] + report["OA_count"] + report["PE_PA_count"] + report["lyso_count"] == 60
    )
    assert report["lyso_count"] == 6 and report["neutral_formula_missing_count"] == 0
    assert report["bis_phosphatidyl_count"] == 0 and report["training_performed"] is False
    assert sum(report["total_carbon_distribution"].values()) == 60
    assert sum(report["total_double_bond_distribution"].values()) == 60
    assert len(report["top_20_least_represented_polar_moieties"]) == 4
    for name in (
        "chain_library",
        "chain_combinations",
        "polar_library",
        "structures",
        "rejections",
    ):
        assert pq.read_table(output / f"{name}.parquet") is not None
    assert load_build_config(output / "build_config_resolved.yaml") == config
    assert json.loads((output / "library_report.json").read_text())["status"] == "complete"


def test_same_seed_reproduces_ids_sampling_and_all_structure_rows(inputs, tmp_path):
    _, _, config = inputs
    rows = []
    for label in ("first", "second"):
        output = tmp_path / label
        build_library(
            config,
            EXAMPLES / "chains.tsv",
            EXAMPLES / "polar_moieties.tsv",
            output,
            EXAMPLES / "p_to_o_mapping.tsv",
        )
        rows.append(pq.read_table(output / "structures.parquet").to_pylist())
    assert rows[0] == rows[1]


def test_does_not_overwrite_existing_build(inputs, tmp_path):
    import pytest

    _, _, config = inputs
    output = tmp_path / "preserve"
    output.mkdir()
    (output / "existing.txt").write_text("preserve")
    with pytest.raises(ValueError, match="nonempty"):
        build_library(config, EXAMPLES / "chains.tsv", EXAMPLES / "polar_moieties.tsv", output)
    assert (output / "existing.txt").read_text() == "preserve"
