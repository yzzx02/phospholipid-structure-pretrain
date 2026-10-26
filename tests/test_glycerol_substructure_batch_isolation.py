"""One invalid extracted candidate must not discard other source records."""

import gzip
import importlib.util
import json
from pathlib import Path


def test_candidate_validation_error_is_recorded_and_batch_continues(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/collect_observed_glycerol_heads_v03.py"
    spec = importlib.util.spec_from_file_location("glycerol_batch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_policy = module.policy

    def reject_one_candidate(mol, **kwargs):
        if any(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()):
            raise ValueError("radical_excluded")
        return original_policy(mol, **kwargs)

    monkeypatch.setattr(module, "policy", reject_one_candidate)
    source = tmp_path / "sources.json.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        json.dump([{"CID": 1, "SMILES": "OCC(O)COC"}, {"CID": 2, "SMILES": "OCC(O)COCCN"}], stream)
    accepted, rejected, counts, cids = module.process_batch(source)
    assert {r["source_CID"] for r in accepted} == {1}
    assert any(
        r[0] == 2 and "core_validation:ValueError:radical_excluded" in r[2] for r in rejected
    )
    assert counts["source_records"] == 2 and counts["accepted_source_CIDs"] == 1
    assert cids == [1, 2]
