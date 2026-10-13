import json
from dataclasses import replace

import pytest

from structure_pretrain.library_builder.budget import MandatoryBudgetExceeded, allocate_budget
from structure_pretrain.library_builder.chain_combinations import active_anchors
from structure_pretrain.library_builder.pipeline import build_library
from structure_pretrain.library_builder.schemas import HARD_MAX_STRUCTURES

from .conftest import EXAMPLES


def test_mandatory_exceed_budget_stops_with_report(inputs, tmp_path):
    _, _, config = inputs
    output = tmp_path / "too_small"
    with pytest.raises(MandatoryBudgetExceeded):
        build_library(
            replace(config, max_structures=27),
            EXAMPLES / "chains.tsv",
            EXAMPLES / "polar_moieties.tsv",
            output,
        )
    report = json.loads((output / "library_report.json").read_text())
    assert report["mandatory_anchor_count"] == 28
    assert report["max_allowed_eligible_polar_heads"] == 3
    assert not (output / "structures.parquet").exists()


def test_anchor_priority_at_exact_budget(inputs, tmp_path):
    _, _, config = inputs
    report = build_library(
        replace(config, max_structures=28),
        EXAMPLES / "chains.tsv",
        EXAMPLES / "polar_moieties.tsv",
        tmp_path / "exact",
    )
    assert report["total_structures"] == 28
    assert report["lyso_count"] == 0
    assert report["mandatory_unique_structures"] == 28


@pytest.mark.parametrize("cap", [29, 31, 43, 60, 100])
def test_never_exceeds_structure_budget(inputs, tmp_path, cap):
    _, _, config = inputs
    report = build_library(
        replace(config, max_structures=cap),
        EXAMPLES / "chains.tsv",
        EXAMPLES / "polar_moieties.tsv",
        tmp_path / str(cap),
        EXAMPLES / "p_to_o_mapping.tsv",
    )
    assert report["total_structures"] <= cap <= HARD_MAX_STRUCTURES


def test_configuration_hard_limit_and_bis_not_silently_enabled(inputs):
    _, _, config = inputs
    with pytest.raises(ValueError):
        replace(config, max_structures=HARD_MAX_STRUCTURES + 1).validate()
    with pytest.raises(NotImplementedError):
        replace(config, include_bis_phosphatidyl=True).validate()


def test_uniform_extra_quotas(inputs):
    chains, polars, config = inputs
    plan = allocate_budget(list(polars.values()), active_anchors(chains, config, []), config)
    assert set(plan.extra_quotas.values()) == {8}
    assert sum(plan.lyso_quotas.values()) == 6
