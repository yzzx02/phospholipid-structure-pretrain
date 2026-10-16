import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import (
    active_anchors,
    build_combination_bank,
)
from structure_pretrain.library_builder.chain_library import import_chains
from structure_pretrain.library_builder.family_sampling import (
    allocate_family_budget,
    dry_run_family_schedule,
    planned_extras,
    weighted_capped_totals,
)
from structure_pretrain.library_builder.full_library_v02 import (
    AnchorOptions,
    build_v02,
    load_config_v02,
    plan_anchors,
)
from structure_pretrain.library_builder.polar_library import import_polars
from structure_pretrain.library_builder.sampling import StratifiedSampler
from structure_pretrain.library_builder.schemas import BuildConfig, load_build_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/library_builder"
WEIGHTS = {"phosphate": 2.0, "phosphonate": 1.0, "polyphosphate": 1.0}


@pytest.fixture
def inputs():
    errors = []
    chains = import_chains(EXAMPLES / "chains.tsv", errors)
    polars = import_polars(EXAMPLES / "polar_moieties.tsv", errors)
    config = replace(
        load_build_config(EXAMPLES / "demo_build.yaml"),
        schema_version="structure-library-v0.2",
        max_structures=60,
        max_extra_per_head=100,
    )
    options = AnchorOptions(family_sampling_weight=WEIGHTS)
    details = {
        p.polar_id: {
            "chemistry_family": "phosphonate" if p.polar_family == "phosphonate" else "phosphate",
            "structure_origin": "synthetic_fixture",
            "is_special_case": p.polar_family == "PE",
        }
        for p in polars
    }
    selected, _ = plan_anchors(polars, active_anchors(chains, config, []), config, options)
    bank = build_combination_bank(chains, config)
    assert not errors
    return chains, polars, details, selected, bank, config, options


def test_v02_two_million_cap_does_not_raise_v01_cap():
    BuildConfig(schema_version="structure-library-v0.2", max_structures=2_000_000).validate()
    with pytest.raises(ValueError):
        BuildConfig(schema_version="structure-library-v0.2", max_structures=2_000_001).validate()
    with pytest.raises(ValueError):
        BuildConfig(max_structures=1_500_001).validate()
    config, options = load_config_v02(ROOT / "configs/library_build_expanded_v02.yaml")
    assert config.max_structures == 2_000_000 and options.family_sampling_weight == WEIGHTS


@pytest.mark.parametrize("maximum", [12, 13, 20, 60, 97, 1500000, 2000000])
def test_integer_quotas_respect_both_caps_and_keep_every_head(inputs, maximum):
    _, polars, details, selected, _, config, _ = inputs
    config = replace(config, max_structures=maximum)
    plan = allocate_family_budget(polars, details, selected, config, WEIGHTS)
    assert plan.mandatory_count == 12
    assert plan.mandatory_count + sum(plan.extra_quotas.values()) <= maximum
    assert set(plan.extra_quotas) == {p.polar_id for p in polars}
    assert max(plan.extra_quotas.values()) <= config.max_extra_per_head
    assert all(0 <= plan.lyso_quotas[i] <= q for i, q in plan.extra_quotas.items())


def test_two_to_one_is_for_extras_not_total_including_mandatory_anchors():
    counts = {"phosphate": 39591, "phosphonate": 62867, "polyphosphate": 939}
    extra = weighted_capped_totals(1_500_000 - sum(counts.values()) * 3, counts, WEIGHTS, 100, 42)
    ratio = (extra["phosphate"] / counts["phosphate"]) / (
        extra["phosphonate"] / counts["phosphonate"]
    )
    assert ratio == pytest.approx(2.0, rel=1e-4)
    full_phosphate = counts["phosphate"] * 3 + extra["phosphate"]
    full_phosphonate = counts["phosphonate"] * 3 + extra["phosphonate"]
    assert full_phosphate > full_phosphonate
    with_seven = weighted_capped_totals(
        1_500_000 - sum(counts.values()) * 7, counts, WEIGHTS, 100, 42
    )
    assert (
        counts["phosphate"] * 7 + with_seven["phosphate"]
        < counts["phosphonate"] * 7 + with_seven["phosphonate"]
    )


def test_invalid_or_zero_weights_never_silently_drop_phosphonate():
    with pytest.raises(ValueError, match="positive"):
        weighted_capped_totals(100, {"phosphonate": 5}, {"phosphonate": 0}, 100, 42)
    with pytest.raises(ValueError, match="Unknown"):
        weighted_capped_totals(100, {"invented": 5}, WEIGHTS, 100, 42)


def test_capped_weighted_waterfill_reallocates_budget():
    result = weighted_capped_totals(
        50, {"phosphate": 1, "phosphonate": 4}, {"phosphate": 1000, "phosphonate": 1}, 10, 42
    )
    assert result == {"phosphate": 10, "phosphonate": 40}


def test_sampling_is_reproducible_without_duplicate_complete_graphs(inputs):
    chains, polars, details, selected, bank, config, _ = inputs
    plan = allocate_family_budget(polars, details, selected, config, WEIGHTS)
    sampler = StratifiedSampler(bank, config)
    by_id = {c.combination_id: c for c in bank}
    canonical = set()
    for polar in polars:
        combos = [by_id[i] for i in sorted(selected[polar.polar_id])]
        combos += list(planned_extras(polar, selected[polar.polar_id], plan, sampler))
        assert len(combos) == len({c.combination_id for c in combos})
        assert list(planned_extras(polar, selected[polar.polar_id], plan, sampler)) == combos[3:]
        for combo in combos:
            key = assemble(chains, combo, polar)["canonical_smiles"]
            assert key not in canonical
            canonical.add(key)
    assert len(canonical) == config.max_structures
    first, _, _ = dry_run_family_schedule(polars, details, selected, bank, config, WEIGHTS)
    second, _, _ = dry_run_family_schedule(
        list(reversed(polars)), details, selected, bank, config, WEIGHTS
    )
    assert first == second
    assert first["estimated_full_structures"] == config.max_structures
    assert first["all_special_heads_retained"]
    assert first["estimated_PE_PA"] > 0 and first["estimated_OA"] > 0


def test_structure_origin_does_not_affect_sampling(inputs):
    _, polars, details, selected, _, config, _ = inputs
    first = allocate_family_budget(polars, details, selected, config, WEIGHTS)
    altered = {k: dict(v, structure_origin="arbitrary_other_source") for k, v in details.items()}
    assert first == allocate_family_budget(polars, altered, selected, config, WEIGHTS)


def test_full_fixture_build_uses_same_family_schedule(inputs, tmp_path, monkeypatch):
    _, polars, details, selected, bank, config, options = inputs
    chains = inputs[0]
    from structure_pretrain.library_builder.full_library_v02 import metadata

    for p in polars:
        family = details[p.polar_id]["chemistry_family"]
        details[p.polar_id] = {
            **metadata({**p.row(), "polar_plausibility_tier": "CORE"}),
            "chemistry_family": family,
            "structure_origin": "synthetic_fixture",
        }
    prepared = (
        {
            "polar_tier_counts": {"CORE": 4},
            "real_filtered_polar_count": 4,
            "generated_supplemental_head_count": 0,
            "adaptive_anchors": False,
            "common_anchor_ids": sorted(selected[polars[0].polar_id]),
            "required_before_optional_rotation": 12,
            "selected_anchor_and_special_structures": 12,
        },
        (chains, polars, details, selected, bank),
    )
    monkeypatch.setattr(
        "structure_pretrain.library_builder.full_library_v02.preflight_v02", lambda *a: prepared
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"status": "fixture"}))
    report = build_v02(
        config,
        options,
        EXAMPLES / "chains.tsv",
        EXAMPLES / "polar_moieties.tsv",
        report_path,
        tmp_path / "output",
    )
    dry, _, _ = dry_run_family_schedule(polars, details, selected, bank, config, WEIGHTS)
    rows = pq.read_table(tmp_path / "output/structures.parquet").to_pylist()
    assert len(rows) == report["total_structures"] == dry["estimated_full_structures"]
    assert report["chemistry_family_used_for_extra_budget"]
    assert report["within_configured_budget"]
    assert report["AA_count"] == dry["estimated_AA"]
    assert report["OA_count"] == dry["estimated_OA"]
    assert report["PE_PA_count"] == dry["estimated_PE_PA"]
    assert report["lyso_count"] == dry["estimated_lyso"]
    assert all("structure_origin" in r and "source_attachment_provenance" in r for r in rows)
