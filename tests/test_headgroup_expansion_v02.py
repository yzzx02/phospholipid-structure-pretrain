import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from structure_pretrain.reported_headgroups.single_anchor_review import analyze_single_anchor
from structure_pretrain.reported_headgroups.source_review_v02 import phosphorus_review_flags


@pytest.mark.parametrize(
    "smiles",
    ["NOP(=O)(O)[*:1]", "OOP(=O)(O)[*:1]", "NP(=O)(O)[*:1]", "OP(=S)(O)[*:1]"],
)
def test_unusual_source_bonds_are_flagged_without_repair(smiles):
    assert phosphorus_review_flags(smiles)


def test_normal_polyphosphate_is_not_a_database_bonding_error():
    assert phosphorus_review_flags("O=P(O)(O)OP(=O)(O)[*:1]") == []


def test_polyaromatic_group_is_not_mislabeled_as_sterol():
    result = analyze_single_anchor("O=P(O)(Oc1cc2ccc3cccc4ccc(c1)c2c34)[*:1]")
    assert not result["sterol_anchor_detected"]
    assert result["auto_status"] != "exclude"


@pytest.mark.parametrize("acyl", ["C(=O)", "C(=O)CCCCCCC"])
def test_second_diacylated_glycerol_arm_excluded_without_class_name(acyl):
    smiles = f"O=P(O)(OCC(O)COP(=O)(O)OCC(OC(=O))CO{acyl})[*:1]"
    result = analyze_single_anchor(smiles)
    assert result["auto_status"] == "exclude"
    assert "extra_diacylated_glycerol_arm" in result["reason_codes"]


def test_branched_isoprenoid_anchor_excluded_but_arginyl_pg_remains_reviewable():
    result = analyze_single_anchor("CC(C)=CCCC(C)=CCOCC(O)COP(=O)(O)[*:1]")
    assert result["longest_hydrocarbon_path"] < 8
    assert result["auto_status"] == "exclude"
    assert "extra_branched_hydrocarbon_anchor" in result["reason_codes"]
    argpg = analyze_single_anchor("NC(N)=NCCCC(N)C(=O)OCC(O)COP(=O)(O)[*:1]")
    assert argpg["auto_status"] == "review"


def csv_file(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("tamper", [None, "decision", "graph", "duplicate_id"])
def test_review_is_applied_by_id_and_source_hash_without_resurrecting_rejection(
    tmp_path, monkeypatch, tamper
):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "expand_v02", scripts / "expand_reported_headgroups_v02.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous, output = tmp_path / "old", tmp_path / "new"
    previous.mkdir()
    output.mkdir()
    heads = [
        {
            "headgroup_id": "fixture-1",
            "polar_smiles": "NCCOP(=O)(O)[*:1]",
            "builder_compatible": "True",
        },
        {
            "headgroup_id": "fixture-2",
            "polar_smiles": "COP(=O)(O)[*:1]",
            "builder_compatible": "True",
        },
    ]
    csv_file(previous / "reported_headgroups_v01.csv", heads)
    for name in ("headgroup_sources.csv", "headgroup_name_alias_mapping.csv"):
        csv_file(
            previous / name,
            [{"headgroup_id": h["headgroup_id"], "name": h["headgroup_id"]} for h in heads],
        )
    review = {
        "schema": "headgroup-review-v01",
        "input_sha256": hashlib.sha256(
            (previous / "reported_headgroups_v01.csv").read_bytes()
        ).hexdigest(),
        "decisions": {
            "fixture-1": {"decision": "keep", "note": "selected"},
            "fixture-2": {"decision": "exclude"},
        },
    }
    review_json = tmp_path / "review.json"
    review_json.write_text(json.dumps(review), encoding="utf-8")
    csv_rows = [
        {
            **h,
            "human_decision": "keep" if i == 0 else "exclude",
            "human_note": "selected" if i == 0 else "",
        }
        for i, h in enumerate(heads)
    ]
    if tamper == "decision":
        csv_rows[0]["human_decision"] = "exclude"
    if tamper == "graph":
        csv_rows[0]["polar_smiles"] = "CCCOP(=O)(O)[*:1]"
    if tamper == "duplicate_id":
        csv_rows[1] = dict(csv_rows[0])
    review_csv = tmp_path / "review.csv"
    csv_file(review_csv, csv_rows)
    before = (previous / "reported_headgroups_v01.csv").read_bytes()
    if tamper:
        with pytest.raises(AssertionError):
            module.save_approved(previous, review_json, review_csv, output)
        assert not (output / "approved_v01").exists()
    else:
        module.save_approved(previous, review_json, review_csv, output)
        kept = module.read_csv(output / "approved_v01/approved_headgroups_v01.csv")
        assert len(kept) == 1 and kept[0]["headgroup_id"] == "fixture-1"
        assert kept[0]["human_note"] == "selected"
    assert (previous / "reported_headgroups_v01.csv").read_bytes() == before
