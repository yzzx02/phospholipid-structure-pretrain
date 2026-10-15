import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest

from structure_pretrain.pubchem_polar.cache import ChemistryCache
from structure_pretrain.pubchem_polar.pipeline import analyze_bundle, validate_remote_queries


def test_chemistry_checkpoint_resume_and_policy_invalidation(tmp_path):
    cache = ChemistryCache(tmp_path, {"max_carbon_count": 30})
    bundle = {"component": None, "candidates": [], "rejection": {"reason": "fixture"}}
    cache.put(1, "OP(=O)(O)O", bundle)
    cache.close()
    resumed = ChemistryCache(tmp_path, {"max_carbon_count": 30})
    assert resumed.get(1, "OP(=O)(O)O") == bundle
    assert resumed.get(1, "OP(=O)(O)C") is None
    resumed.close()
    changed = ChemistryCache(tmp_path, {"max_carbon_count": 20})
    assert changed.get(1, "OP(=O)(O)O") is None
    changed.close()


def test_remote_cid_controls_fail_even_when_local_property_controls_pass():
    config = {"queries": [{"query_id": "phosphate", "smarts": "P", "positive_controls": ["PA"]}]}
    live = {"PA": {"passed": True, "records": [{"cid": 1004, "passed": True}]}}

    class Client:
        def query(self, smarts, cap):
            return [1003]

    with pytest.raises(RuntimeError, match="Server-side positive CID controls"):
        validate_remote_queries(Client(), config, live)


def test_remote_cid_controls_confirm_designated_neutral_precursor():
    config = {"queries": [{"query_id": "phosphate", "smarts": "P", "positive_controls": ["PA"]}]}
    live = {"PA": {"passed": True, "records": [{"cid": 1004, "passed": True}]}}

    class Client:
        def query(self, smarts, cap):
            return [1003, 1004]

    assert validate_remote_queries(Client(), config, live)["phosphate"]["passed"]


def test_spawned_chemistry_workers_match_serial_outputs():
    smiles = ["OP(=O)(O)OCCN", "OP(=O)([O-])OCC[N+](C)(C)C", "OP(=S)(O)O"]
    serial = [analyze_bundle(s, None) for s in smiles]
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        parallel = list(pool.map(analyze_bundle, smiles, [None] * len(smiles)))
    assert serial == parallel
