import csv
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from rdkit import Chem

from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.polar_library import import_polars
from structure_pretrain.polar_expansion.chemistry import (
    AnalogIneligible,
    acid_family,
    element_counts,
    enumerate_source,
    evaluate_graph,
    phosphate_analog,
)
from structure_pretrain.polar_expansion.pipeline import expand_polars, merge_analog
from structure_pretrain.polar_plausibility.pipeline import write_tsv
from structure_pretrain.polar_plausibility.reactive_rules import load_rules
from structure_pretrain.pubchem_polar.chemistry import process_smiles

ROOT = Path(__file__).resolve().parents[2]
PLAUSIBILITY = ROOT / "configs/polar_plausibility.yaml"
RULES = ROOT / "configs/polar_reactive_rules.yaml"


@pytest.fixture
def settings():
    return yaml.safe_load(PLAUSIBILITY.read_text()), load_rules(RULES)


def test_one_source_has_two_distinct_valid_fragments_and_equivalent_sites(settings):
    smiles = "OP(=O)(O)OCC(C)OP(=O)(O)O"
    candidates = process_smiles(smiles)["candidates"]
    keys = {c["polar_fragment_smiles"] for c in candidates}
    assert len(keys) == 2
    rows, audit = enumerate_source(123, smiles, keys)
    assert len(rows) == 2 and audit["enumerated_sites"] == 4
    assert all(len(r["sites"]) == 2 for r in rows.values())
    for key in rows:
        assert evaluate_graph(key, *settings)["tier"] in {"CORE", "EXTENDED"}


def test_equivalent_sites_have_one_canonical_fragment_and_both_provenance_indices():
    smiles = "NCCOP(=O)(O)O"
    keys = {c["polar_fragment_smiles"] for c in process_smiles(smiles)["candidates"]}
    rows, _ = enumerate_source(124, smiles, keys)
    assert len(rows) == 1 and len(next(iter(rows.values()))["sites"]) == 2


def test_salt_source_indices_reference_original_raw_graph():
    smiles = "[Na+].NCCOP(=O)([O-])O"
    keys = {c["polar_fragment_smiles"] for c in process_smiles(smiles)["candidates"]}
    rows, _ = enumerate_source(125, smiles, keys)
    original = Chem.MolFromSmiles(smiles)
    for row in rows.values():
        for site in row["sites"]:
            atom = original.GetAtomWithIdx(site["source_attachment_atom_index"])
            assert atom.GetAtomicNum() == 8 and atom.GetDegree() == 1
            assert atom.GetNeighbors()[0].GetAtomicNum() == 15
            assert (
                site["source_attachment_atom_index"]
                != site["selected_component_attachment_atom_index"]
            )


def test_reactive_and_builder_boundary_are_not_rescued(settings):
    reactive = evaluate_graph("O=P(O)(OCl)[*:1]", *settings)
    assert reactive["tier"] == "REJECT" and reactive["reactive_flags"]
    boundary = evaluate_graph("CC(=O)OCC(O)COP(=O)(O)[*:1]", *settings)
    assert boundary["tier"] == "HOLD" and boundary["builder_boundary_hold"]


@pytest.mark.parametrize(
    "smiles",
    [
        "NCCP(=O)(O)[*:1]",
        "CCP(=O)(O)[*:1]",
        "N#CCP(=O)(O)[*:1]",
        "O=P(O)(c1ccccc1)[*:1]",
        "O=P(O)(CCS)[*:1]",
        "O=P(O)(CCCl)[*:1]",
    ],
)
def test_analog_inserts_one_oxygen_and_preserves_other_elements(smiles, settings):
    result = phosphate_analog(smiles)
    before = element_counts(Chem.MolFromSmiles(smiles))
    after = element_counts(Chem.MolFromSmiles(result["polar_fragment_smiles"]))
    assert after["O"] == before["O"] + 1
    for e in ("C", "N", "S", "P", "F", "Cl", "Br", "I"):
        assert after[e] == before[e]
    assert acid_family(Chem.MolFromSmiles(result["polar_fragment_smiles"])) == "phosphate"
    assert not evaluate_graph(result["polar_fragment_smiles"], *settings)["evaluation_error"]


def test_aminoethyl_analog_is_exact_pe_connectivity():
    result = phosphate_analog("NCCP(=O)(O)[*:1]")
    expected = Chem.MolToSmiles(Chem.MolFromSmiles("NCCOP(=O)(O)[*:1]"))
    assert result["polar_fragment_smiles"] == expected


def test_analog_newly_matching_builder_boundary_is_held_not_a_crash(settings):
    result = phosphate_analog("CC(=O)OCC(O)CP(=O)(O)[*:1]")
    classified = evaluate_graph(result["polar_fragment_smiles"], *settings)
    assert classified["tier"] == "HOLD" and classified["builder_boundary_hold"]


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("O=P1([*:1])OCCC1", "ANALOG_RING_PC_BOND"),
        ("O=P(O)(CCP(=O)(O)O)[*:1]", "ANALOG_MULTI_P"),
        ("O=P(O)(OP(=O)(O)O)[*:1]", "ANALOG_MULTI_P"),
        ("NCCOP(=O)(O)[*:1]", "ANALOG_NOT_PHOSPHONATE"),
    ],
)
def test_ineligible_analogs_are_explicit(smiles, reason):
    with pytest.raises(AnalogIneligible, match=reason):
        phosphate_analog(smiles)


def test_analog_dedup_keeps_original_id_and_does_not_claim_parent_cid_as_target_source(settings):
    parent = {"polar_id": "AEP", "source_cids": [777]}
    analog = phosphate_analog("NCCP(=O)(O)[*:1]")
    key = analog["polar_fragment_smiles"]
    existing = {
        "polar_id": "PE",
        "chemistry_family": "phosphate",
        "source_cids": [888],
        "analog_parent_polar_ids": [],
        "parent_source_cids": [],
    }
    rows = {key: existing}
    is_new, identifier = merge_analog(rows, parent, analog, evaluate_graph(key, *settings))
    assert not is_new and identifier == "PE" and len(rows) == 1
    assert existing["source_cids"] == [888] and existing["parent_source_cids"] == [777]
    rows = {}
    assert merge_analog(rows, parent, analog, evaluate_graph(key, *settings))[0]
    generated = rows[key]
    assert generated["source_cids"] == [] and generated["source_count"] == 0
    assert generated["parent_source_cids"] == [777]


def test_small_pipeline_preserves_baseline_and_only_rescues_multi(tmp_path):
    review, output = tmp_path / "review", tmp_path / "expanded"
    review.mkdir()
    baseline = []
    for identifier, smiles, family in [
        ("PE", "NCCOP(=O)(O)[*:1]", "phosphate"),
        ("AEP", "NCCP(=O)(O)[*:1]", "phosphonate"),
        ("EP", "CCP(=O)(O)[*:1]", "phosphonate"),
    ]:
        baseline.append(
            {
                "polar_id": identifier,
                "source": "synthetic_fixture",
                "source_id": "fixture",
                "original_smiles": "",
                "polar_fragment_smiles": smiles,
                "attachment_atom_map": 1,
                "polar_family": "PE" if identifier == "PE" else "phosphonate",
                "allow_aa": True,
                "allow_oa": True,
                "allow_pa": identifier == "PE",
                "chemistry_family": family,
                "polar_plausibility_tier": "CORE",
                "source_cids": [900000001],
                "source_count": 1,
            }
        )
    baseline_path = tmp_path / "baseline.tsv"
    write_tsv(baseline_path, baseline, list(baseline[0]))
    frozen_bytes = baseline_path.read_bytes()
    source = "OP(=O)(O)OP(=O)(O)OCCN"
    candidates = {c["polar_fragment_smiles"]: c for c in process_smiles(source)["candidates"]}
    fragments = [
        dict(c, polar_id=f"MULTI_{i}", source_cids=[900000002])
        for i, c in enumerate(candidates.values())
    ]
    fragments += [
        dict(
            fragments[0],
            polar_id="NO_RESCUE",
            status="CHARGE_AMBIGUOUS",
            polar_fragment_smiles="unprocessed_invalid",
            source_cids=[900000003],
        )
    ]
    pq.write_table(pa.Table.from_pylist(fragments), review / "pubchem_polar_fragments.parquet")
    pq.write_table(
        pa.Table.from_pylist([{"cid": 900000002, "original_smiles": source}]),
        review / "pubchem_polar_raw.parquet",
    )
    config_path = tmp_path / "expansion.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "workers": 1,
                "baseline_sha256": file_hash(baseline_path),
                "baseline_family_counts": {"phosphate": 1, "phosphonate": 2},
                "allowed_rescue_status": "MULTIPLE_ATTACHMENT_AMBIGUITY",
            }
        )
    )
    report = expand_polars(review, baseline_path, output, config_path, PLAUSIBILITY, RULES)
    assert report["rescued_new_unique_by_family"] == {"polyphosphate": 2}
    assert report["expanded_family_counts"] == {
        "phosphate": 2,
        "phosphonate": 2,
        "polyphosphate": 2,
    }
    assert report["analog_counts"]["duplicate_existing_phosphate"] == 1
    assert report["analog_counts"]["new_unique_phosphate"] == 1
    assert report["charge_or_filter_review_rescued"] == 0
    assert report["phosphonate_heads_removed_for_balance"] == 0
    assert baseline_path.read_bytes() == frozen_bytes
    errors = []
    assert len(import_polars(output / "polar_moieties_expanded_v0.2.tsv", errors)) == 6
    assert not errors
    with (output / "polar_moieties_expanded_v0.2.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert {r["polar_id"] for r in baseline}.issubset(r["polar_id"] for r in rows)
    assert Counter(r["structure_origin"] for r in rows)["generated_phosphate_analog"] == 1
