from pathlib import Path

import pyarrow.parquet as pq
import yaml

from structure_pretrain.pubchem_polar.chemistry import read_controls, validate_controls
from structure_pretrain.pubchem_polar.pipeline import process_records

ROOT = Path(__file__).resolve().parents[2]


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
