import json
import runpy
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from structure_pretrain.pubchem_polar.cache import ChemistryCache, seed_from_review
from structure_pretrain.pubchem_polar.chemistry import read_controls, validate_controls
from structure_pretrain.pubchem_polar.pipeline import process_records

ROOT = Path(__file__).resolve().parents[2]


def test_empty_or_failed_query_has_valid_empty_review_outputs(tmp_path):
    result = process_records(None, {}, [], tmp_path, {}, {})
    assert result["properties_downloaded"] == result["unique_fragments"] == 0
    assert result["missing_property_cids"] == []
    for name in (
        "polar_raw",
        "p_components",
        "attachment_candidates",
        "polar_fragments",
        "polar_rejections",
    ):
        assert pq.read_table(tmp_path / f"pubchem_{name}.parquet").num_rows == 0


def test_parquet_review_provenance_and_rejections(tmp_path):
    config = yaml.safe_load((ROOT / "configs/pubchem_polar_queries.yaml").read_text())
    _, known = validate_controls(config, read_controls(ROOT / "configs/known_head_controls.tsv"))
    inputs = {1: "OP(=O)(O)OCCN", 2: "[Na+].[O-]P(=O)(O)OCCN", 3: "COP(=O)(OC)OC"}

    class Client:
        def get_properties(self, cids):
            return [
                {
                    "CID": cid,
                    "ConnectivitySMILES": inputs[cid],
                    "SMILES": inputs[cid],
                    "MolecularFormula": "fixture",
                    "MolecularWeight": "141.1",
                    "Charge": 0,
                }
                for cid in cids
            ]

    matches = {cid: {"phosphate_acid"} for cid in inputs}
    result = process_records(Client(), matches, config["queries"], tmp_path, config, known)
    assert result["properties_downloaded"] == 3
    assert result["single_p_component_records"] == 3
    assert result["attachment_eligible_cids"] == 2
    assert result["counterions_removed"] == 1
    assert result["unique_fragments"] == 1
    assert result["rejections"] == {"no_acidic_attachment": 1}
    rows = pq.read_table(tmp_path / "pubchem_polar_fragments.parquet").to_pylist()
    assert rows[0]["source_cids"] == [1, 2] and rows[0]["source_count"] == 2
    assert rows[0]["allow_pa"] and rows[0]["attachment_site_count_before_dedup"] == 4
    assert pq.read_table(tmp_path / "pubchem_p_components.parquet").num_rows == 3
    assert pq.read_table(tmp_path / "pubchem_polar_rejections.parquet").num_rows == 1
    assert (tmp_path / "pubchem_polar_review.tsv").exists()
    assert not (tmp_path / "polar_moieties_pubchem_v0.1.tsv").exists()
    report = {
        **result,
        "unique_cids": 3,
        "properties_complete": True,
        "query_results": [{"query_id": "phosphate_acid", "cid_count": 3}],
    }
    (tmp_path / "pubchem_polar_report.json").write_text(json.dumps(report), encoding="utf-8")
    audit = runpy.run_path(str(ROOT / "scripts/audit_pubchem_polar.py"))["audit"]
    assert audit(tmp_path)["passed"]
    report["pubchem_query_controls"] = {"phosphate_acid": {"expected_cids": [999]}}
    (tmp_path / "pubchem_polar_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Designated server-side positive CID"):
        audit(tmp_path)
    report["pubchem_query_controls"] = {"phosphate_acid": {"expected_cids": [1]}}
    (tmp_path / "pubchem_polar_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert audit(tmp_path)["designated_server_control_pairs_verified"] == 1
    state = tmp_path / "state"
    state.mkdir()
    cache = ChemistryCache(state, config["limits"])
    report["chemistry_policy_sha256"] = cache.policy_hash
    (tmp_path / "pubchem_polar_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert seed_from_review(cache, tmp_path) == 3
    assert len(cache.get(1, inputs[1])["candidates"]) == 2
    assert cache.get(3, inputs[3])["rejection"]["reason"] == "no_acidic_attachment"
    assert "matched_query_ids" not in cache.get(1, inputs[1])["candidates"][0]
    cache.close()
    # A provenance error in persisted data must be found independently of pipeline counters.
    report["pubchem_query_controls"] = {}
    (tmp_path / "pubchem_polar_report.json").write_text(json.dumps(report), encoding="utf-8")
    raw_path = tmp_path / "pubchem_polar_raw.parquet"
    raw = pq.read_table(raw_path)
    pq.write_table(raw.slice(1), raw_path)
    with pytest.raises(ValueError, match="Raw row count"):
        audit(tmp_path)
