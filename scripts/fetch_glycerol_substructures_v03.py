"""Complete a declared PubChem scaffold search with compact, resumable batches."""

import argparse
import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

from structure_pretrain.glycerol_core_collection.collection import dump_json, file_hash

CONFIG = Path("configs/glycerol_substructure_collection_v03_20261026.json")
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
LOCK = threading.Lock()
LAST = 0.0
LOCAL = threading.local()


def request(url, *, params=None, data=None):
    global LAST
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.proxies.update(
            {"http": "http://127.0.0.1:17891", "https": "http://127.0.0.1:17891"}
        )
    for attempt in range(4):
        with LOCK:
            time.sleep(max(0.0, 0.55 - (time.monotonic() - LAST)))
            LAST = time.monotonic()
        try:
            response = LOCAL.session.request(
                "POST" if data else "GET", url, params=params, data=data, timeout=150
            )
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(30, max(int(response.headers.get("Retry-After", "0")), 2**attempt)))
                continue
            response.raise_for_status()
            value = response.json()
            if "Fault" in value:
                raise ValueError(str(value["Fault"]))
            return value
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("PubChem request retries exhausted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text("utf-8"))
    root = Path(config["root"])
    raw = root / "raw"
    nonp = bool(config.get("new_PubChem_branch_requires_no_P"))
    batches = raw / ("nonP_properties" if nonp else "properties")
    batches.mkdir(parents=True, exist_ok=True)
    search_manifest = raw / "scaffold_search_manifest.json"
    ids_path = raw / "scaffold_cids.npy"
    if ids_path.exists():
        ids = np.load(ids_path)
        manifest = json.loads(search_manifest.read_text("utf-8"))
        assert manifest["ids_SHA256"] == file_hash(ids_path)
        assert len(ids) == manifest["retrieved_CIDs"] < config["query_MaxRecords"]
    else:
        url = (
            PUG
            + "/compound/fastsubstructure/smarts/"
            + quote(config["source_query_smarts"], safe="")
            + "/cids/JSON"
        )
        params = {
            "MaxRecords": config["query_MaxRecords"],
            "ChainsMatchRings": "false",
            "Stereo": "ignore",
            "MaxSeconds": 120,
        }
        result = request(url, params=params)
        ids = np.array(sorted(set(result["IdentifierList"]["CID"])), dtype=np.int64)
        assert len(ids) < config["query_MaxRecords"], "query truncated; partition before publishing"
        probe_path = raw / "glycerol_smarts_probe.json.gz"
        if probe_path.exists():
            with gzip.open(probe_path, "rt", encoding="utf-8") as stream:
                expected = json.load(stream)["IdentifierList"]["Size"]
            assert len(ids) == expected, "probe count and retrieved CIDs differ"
        np.save(ids_path, ids)
        manifest = {
            "source": "PubChem PUG REST",
            "query_url": url,
            "parameters": params,
            "retrieved_CIDs": len(ids),
            "not_at_MaxRecords_cap": True,
            "probe_Size_matches_retrieved": probe_path.exists(),
            "ids_SHA256": file_hash(ids_path),
            "config_SHA256": file_hash(CONFIG),
            "retrieved_UTC": datetime.now(timezone.utc).isoformat(),
            "complete_only_for_declared_acyclic_glycerol_3O_query": True,
        }
        dump_json(search_manifest, manifest)
    print(json.dumps({"stage": "scaffold_CIDs", "N": len(ids), "complete_query": True}), flush=True)
    if args.search_only:
        return
    if nonp:
        filtered = np.load(raw / "nonP_scaffold_cids.npy")
        removed = np.load(raw / "phosphorus_source_cids.npy")
        assert np.array_equal(np.union1d(filtered, removed), ids)
        assert not len(np.intersect1d(filtered, removed))
        manifest.update(
            all_glycerol_source_count=len(ids),
            excluded_P_source_count=len(removed),
            candidate_nonP_count=len(filtered),
            latest_config_SHA256=file_hash(CONFIG),
        )
        ids = filtered
    cached = {}
    needed = set(ids.tolist())
    # Reuse properties fetched before the user narrowed the new branch to nonP.
    if nonp:
        for path in (raw / "properties").glob("properties_*.json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for r in json.load(stream):
                    if r["CID"] in needed:
                        cached[r["CID"]] = r
        print(
            json.dumps({"stage": "reuse_cached_nonP", "N": len(cached), "nonP_total": len(ids)}),
            flush=True,
        )
    batch_size = int(config["property_batch_size"])
    jobs = [(i, ids[i : i + batch_size]) for i in range(0, len(ids), batch_size)]

    def fetch(job):
        offset, group = job
        target = batches / f"properties_{offset:07d}.json.gz"
        if target.exists():
            with gzip.open(target, "rt", encoding="utf-8") as stream:
                records = json.load(stream)
        else:
            records = [cached[int(cid)] for cid in group if int(cid) in cached]
            missing = [int(cid) for cid in group if int(cid) not in cached]
            if missing:
                result = request(
                    PUG + "/compound/cid/property/SMILES,ConnectivitySMILES,MolecularFormula,"
                    "MolecularWeight,IUPACName/JSON",
                    data={"cid": ",".join(map(str, missing))},
                )
                records.extend(result["PropertyTable"]["Properties"])
            assert {r["CID"] for r in records} == set(group.tolist())
            with gzip.open(target.with_suffix(".gz.part"), "wt", encoding="utf-8") as stream:
                json.dump(records, stream, ensure_ascii=False, separators=(",", ":"))
            target.with_suffix(".gz.part").replace(target)
        assert {r["CID"] for r in records} == set(group.tolist())
        return len(records)

    completed = 0
    with ThreadPoolExecutor(max_workers=int(config["network_workers"])) as executor:
        for count in executor.map(fetch, jobs):
            completed += count
            if completed % (batch_size * 5) == 0:
                print(
                    json.dumps({"stage": "properties", "completed": completed, "total": len(ids)}),
                    flush=True,
                )
    assert completed == len(ids)
    files = list(batches.glob("properties_*.json.gz"))
    manifest.update(
        properties_complete=True,
        properties_count=completed,
        batches=len(files),
        compressed_property_bytes=sum(p.stat().st_size for p in files),
    )
    dump_json(raw / "fetch_complete.json", manifest)
    print(
        json.dumps(
            {
                "stage": "fetch_complete",
                "N": completed,
                "compressed_MB": manifest["compressed_property_bytes"] / 1e6,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
