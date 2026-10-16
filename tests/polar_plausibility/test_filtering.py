import csv
import json
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from structure_pretrain.library_builder.polar_library import import_polars
from structure_pretrain.polar_plausibility.pipeline import filter_snapshot
from structure_pretrain.polar_plausibility.reactive_rules import load_rules
from structure_pretrain.polar_plausibility.scaffold import scaffold_id
from structure_pretrain.polar_plausibility.tiering import assign_tier, evaluate
from structure_pretrain.pubchem_polar.chemistry import process_smiles, read_controls

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/polar_plausibility.yaml"
RULES = ROOT / "configs/polar_reactive_rules.yaml"
CONTROLS = ROOT / "configs/known_head_controls.tsv"


@pytest.fixture
def settings():
    return yaml.safe_load(CONFIG.read_text()), load_rules(RULES)


def checked(smiles, settings):
    config, rules = settings
    return evaluate({"status": "APPROVABLE", "polar_fragment_smiles": smiles}, config, rules)


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("O=P(O)(OF)[*:1]", "REJECT_REACTIVE_O_HALOGEN"),
        ("O=P(O)(OCl)[*:1]", "REJECT_REACTIVE_O_HALOGEN"),
        ("O=P(O)(CC(=O)Cl)[*:1]", "REJECT_ACYL_HALIDE"),
        ("O=P(O)(CCS(=O)(=O)Cl)[*:1]", "REJECT_SULFONYL_HALIDE"),
        ("O=P(O)(OOC)[*:1]", "REJECT_PEROXIDE"),
    ],
)
def test_reactive_rejects(smiles, reason, settings):
    row = checked(smiles, settings)
    assert row["tier"] == "REJECT" and reason in row["reactive_flags"]


@pytest.mark.parametrize(
    "smiles",
    [
        "O=P(O)(OP(=O)(O)O)[*:1]",
        "NCCOP(=O)(O)[*:1]",
        "C[N+](C)(C)CCOP(=O)([O-])[*:1]",
        "NCCP(=O)(O)[*:1]",
        "O=C(O)CP(=O)(O)[*:1]",
        "O=P(O)(OC1OC(O)C(O)C(O)C1O)[*:1]",
        "O=P(O)(OCCn1ccnc1)[*:1]",
        "C#CCP(=O)(O)[*:1]",
        "O=P(O)(c1ccccc1)[*:1]",
        "O=P(O)(OCCCl)[*:1]",
        "O=P(O)(OC1C(O)C(O)C(O)C(O)C1O)[*:1]",
    ],
)
def test_unknown_chemistry_is_retained(smiles, settings):
    row = checked(smiles, settings)
    assert row["tier"] in {"CORE", "EXTENDED"} and not row["reactive_flags"]
    assert not row["evaluation_error"]


def test_extended_and_low_hetero_hold(settings):
    assert checked("O=P(O)(" + "OC" * 10 + ")[*:1]", settings)["tier"] == "EXTENDED"
    row = checked("O=P(O)(c1ccc(-c2ccc(-c3ccc(-c4ccc(-c5ccccc5)cc4)cc3)cc2)cc1)[*:1]", settings)
    assert row["hetero_fraction"] < 0.1 and row["tier"] == "HOLD"


def test_dummy_is_not_a_heavy_atom_and_round_trip_failure_is_reported(settings):
    row = checked("NCCOP(=O)(O)[*:1]", settings)
    assert row["heavy_atom_count"] == 7 and row["hetero_count"] == 5
    bad = checked("not_smiles", settings)
    assert bad["tier"] == "REJECT" and bad["evaluation_error"]


def test_all_positive_controls_core_without_rescue(settings):
    for control in read_controls(CONTROLS):
        if control["kind"] == "positive":
            for fragment in process_smiles(control["smiles"])["candidates"]:
                assert checked(fragment["polar_fragment_smiles"], settings)["tier"] == "CORE"


def test_sensitivity_changes_tiers_not_structures(settings):
    config, _ = settings
    row = checked("O=P(O)(" + "OC" * 10 + ")[*:1]", settings)
    alternative = deepcopy(config)
    alternative["core"]["max_molecular_weight"] = 1000
    assert assign_tier(row, [], alternative)[0] in {"CORE", "EXTENDED"}
    assert len(scaffold_id(row["polar_fragment_smiles"])) > 60


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    fragments, raw = [], []
    for cid, control in enumerate(read_controls(CONTROLS), 1):
        if control["kind"] != "positive":
            continue
        processed = process_smiles(control["smiles"])
        unique = {c["polar_fragment_smiles"]: c for c in processed["candidates"]}
        for candidate in unique.values():
            fragments.append(
                dict(
                    candidate,
                    polar_id=f"test_{cid}",
                    source_cids=[cid],
                    source_count=1,
                    known_control_ids=[control["control_id"]],
                )
            )
        raw.append({"cid": cid, "original_smiles": control["smiles"]})
    # Non-APPROVABLE input must remain entirely outside this layer.
    fragments.append(dict(fragments[0], polar_id="excluded", status="FILTER_REVIEW"))
    pq.write_table(pa.Table.from_pylist(fragments), root / "pubchem_polar_fragments.parquet")
    pq.write_table(pa.Table.from_pylist(raw), root / "pubchem_polar_raw.parquet")
    (root / "pubchem_polar_report.json").write_text('{"properties_complete": false}')
    return root


def test_complete_pipeline_is_builder_compatible_reproducible_and_offline(snapshot, tmp_path):
    tables = []
    for label in ("one", "two"):
        output = tmp_path / label
        report = filter_snapshot(snapshot, output, CONFIG, RULES, CONTROLS)
        assert report["input_approvable_count"] == 9
        assert report["tier_counts"]["CORE"] == report["approved_polar_count"] == 9
        assert not report["training_performed"] and report["network_requests"] == 0
        assert not report["full_lipid_library_generated"]
        approved = output / "polar_moieties_pubchem_v0.1.tsv"
        errors = []
        assert len(import_polars(approved, errors)) == 9 and not errors
        with approved.open(newline="", encoding="utf-8") as stream:
            exported = list(csv.DictReader(stream, delimiter="\t"))
        assert sum(row["allow_pa"] == "True" for row in exported) == 1
        assert next(row for row in exported if row["allow_pa"] == "True")["polar_family"] == "PE"
        scaffold = json.loads((output / "polar_scaffold_report.json").read_text())
        assert scaffold["fragment_count"] == 9
        assert sum(scaffold["scaffold_counts"].values()) == 9
        tables.append(pq.read_table(output / "polar_plausibility_all.parquet").to_pylist())
    assert tables[0] == tables[1]


def test_second_layer_never_accepts_other_review_status(settings):
    config, rules = settings
    with pytest.raises(ValueError, match="APPROVABLE"):
        evaluate({"status": "CHARGE_AMBIGUOUS"}, config, rules)


def test_existing_builder_boundary_goes_to_hold_without_chemical_rejection(settings):
    row = checked("CC(=O)OCC(O)COP(=O)(OC1C(O)C(O)C(O)C(O)C1O)[*:1]", settings)
    assert row["tier"] == "HOLD" and row["builder_boundary_hold"]
    assert not row["reactive_flags"] and not row["evaluation_error"]
