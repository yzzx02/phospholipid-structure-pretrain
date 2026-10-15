from copy import deepcopy

import pyarrow.parquet as pq

from structure_pretrain.pubchem_polar.chemistry import process_smiles
from structure_pretrain.pubchem_polar.fragment_store import FragmentStore
from structure_pretrain.pubchem_polar.pipeline import (
    FRAGMENT_SCHEMA,
    merge_candidate,
    process_records,
)


def test_sqlite_aggregation_matches_memory_provenance_flags_and_sites(tmp_path):
    candidate = process_smiles("OP(=O)(O)OCCN")["candidates"][0]
    control = {"PE": [candidate["polar_fragment_smiles"]]}
    memory, store = {}, FragmentStore(tmp_path, FRAGMENT_SCHEMA.names, ["a", "b"])
    changed = deepcopy(candidate)
    changed.update(status="FILTER_REVIEW", filter_flags=["molecular_weight"])
    for item, cid, queries in (
        (candidate, 20, ["b"]),
        (candidate, 10, ["a"]),
        (changed, 20, ["a", "b"]),
    ):
        merge_candidate(memory, item, cid, queries, control)
        merge_candidate(store, item, cid, queries, control)
    rows = list(store.iter_rows())
    assert len(store) == len(rows) == 1
    assert candidate["polar_fragment_smiles"] in store
    expected = next(iter(memory.values()))
    for key in ("source_cids", "matched_query_ids", "filter_flags", "review_flags"):
        expected[key] = sorted(expected[key])
    expected["source_count"] = len(expected["source_cids"])
    assert rows == [expected]
    assert rows[0]["attachment_site_count_before_dedup"] == 3
    assert rows[0]["source_cids"] == [10, 20] and rows[0]["allow_pa"]
    store.close()
    assert not list(tmp_path.glob("polar-fragments-*"))


def test_streamed_disk_review_equals_in_memory_outputs(tmp_path):
    smiles = {
        1: "OP(=O)(O)OCCN",
        2: "[Na+].[O-]P(=O)(O)OCCN",
        3: "COP(=O)(OC)OC",
        4: "OP(=O)(O)OCCOP(=O)(O)OC",
    }

    class Client:
        def get_properties(self, cids):
            return [
                {
                    "CID": cid,
                    "SMILES": smiles[cid],
                    "ConnectivitySMILES": smiles[cid],
                    "MolecularFormula": "fixture",
                    "MolecularWeight": "140",
                    "Charge": 0,
                }
                for cid in cids
            ]

    class DiskClient(Client):
        cache_dir = tmp_path / "cache"

    DiskClient.cache_dir.mkdir()
    memory, disk = tmp_path / "memory", tmp_path / "disk"
    memory.mkdir()
    disk.mkdir()
    config = {"chemistry_workers": 1, "api": {"property_batch_size": 2}}
    matches = {cid: {"q"} for cid in smiles}
    queries = [{"query_id": "q", "family": "phosphate"}]
    a = process_records(Client(), matches, queries, memory, config, {})
    b = process_records(DiskClient(), matches, queries, disk, config, {})
    for name in (
        "polar_raw",
        "p_components",
        "attachment_candidates",
        "polar_fragments",
        "polar_rejections",
    ):
        assert pq.read_table(memory / f"pubchem_{name}.parquet").equals(
            pq.read_table(disk / f"pubchem_{name}.parquet")
        )
    for field in (
        "unique_fragments",
        "family_counts",
        "status_counts",
        "distributions",
        "charge_review_breakdown",
        "representative_expanded_fragments_outside_controls",
    ):
        assert a[field] == b[field]
    assert not list(DiskClient.cache_dir.glob("polar-fragments-*"))
