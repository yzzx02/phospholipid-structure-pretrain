"""Bounded official PubChem fetch with compact G-drive caches and explicit truncation."""

import gzip
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlencode

import requests

CONFIG = Path("configs/glycerol_core_collection_v01.json")
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def main():
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = Path(c["root"])
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = []
    queries = []
    membership = {}

    def fetch(url):
        label = hashlib.sha256(url.encode()).hexdigest()
        destination = raw / (label + ".json.gz")
        if destination.exists():
            with gzip.open(destination, "rt", encoding="utf-8") as stream:
                data = json.load(stream)
            status = "cache"
        else:
            session = requests.Session()
            session.proxies.update(
                {"http": "http://127.0.0.1:17891", "https": "http://127.0.0.1:17891"}
            )
            for attempt in range(3):
                try:
                    response = session.get(url, timeout=60)
                    if response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(1 + attempt)
                        continue
                    data = response.json()
                    status = response.status_code
                    with gzip.open(destination, "wt", encoding="utf-8") as stream:
                        json.dump(data, stream, ensure_ascii=False)
                    break
                except (requests.RequestException, ValueError):
                    if attempt == 2:
                        raise
            else:
                raise RuntimeError("Source download failed: " + url)
            time.sleep(0.35)
        manifest.append(
            {
                "url": url,
                "cache": destination.name,
                "status": status,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
        return data

    for kind, names in (
        ("full_lipid", c["full_lipid_names"]),
        ("sugar_parent", c["natural_sugar_building_blocks"]),
    ):
        for name in names:
            result = fetch(PUG + "/compound/name/" + quote(name, safe="") + "/cids/JSON")
            ids = result.get("IdentifierList", {}).get("CID", [])
            if not ids and result.get("Fault", {}).get("Code") != "PUGREST.NotFound":
                raise ValueError("Unexpected name query response: " + name)
            queries.append({"kind": kind, "query": name, "count": len(ids), "cids": ids})
            for cid in ids:
                membership.setdefault(str(cid), []).append({"kind": kind, "name": name})
            print(json.dumps({"stage": "PubChem_name", "name": name, "N": len(ids)}), flush=True)
    for key, term in c["entrez_queries"].items():
        result = fetch(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            + urlencode(
                {
                    "db": "pccompound",
                    "term": term,
                    "retmax": c["network_search_limit_per_query"],
                    "retmode": "json",
                }
            )
        )
        search = result.get("esearchresult")
        if not search:
            raise ValueError("Entrez response missing: " + key)
        ids = search.get("idlist", [])
        queries.append(
            {
                "kind": "text",
                "query": term,
                "key": key,
                "reported_count": int(search["count"]),
                "downloaded_count": len(ids),
                "truncated": int(search["count"]) > len(ids),
                "cids": ids,
            }
        )
        for cid in ids:
            membership.setdefault(str(cid), []).append({"kind": "text", "name": key})
        print(
            json.dumps(
                {
                    "stage": "PubChem_text",
                    "query": key,
                    "total": search["count"],
                    "downloaded": len(ids),
                }
            ),
            flush=True,
        )
    ids = sorted(map(int, membership))
    # Supplement representative DGTS/DGTA/DGCC records with explicit chain-bearing structures.
    for cid in (42607374, 171116410, 56936293, 171116409, 171117775, 178328770):
        membership.setdefault(str(cid), []).append(
            {"kind": "curated_example", "name": "LMSD_cross_reference"}
        )
    ids = sorted(map(int, membership))

    def properties(batch):
        result = fetch(
            PUG
            + "/compound/cid/"
            + ",".join(map(str, batch))
            + "/property/SMILES,ConnectivitySMILES,IUPACName,MolecularFormula,MolecularWeight/JSON"
        )
        if "PropertyTable" not in result:
            raise ValueError("PubChem property batch missing")
        return result["PropertyTable"]["Properties"]

    records = []
    workers = int(c["network_threads"])
    if workers not in (1, 2):
        raise ValueError("Use one or two network workers for this bounded PubChem collection")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch in executor.map(properties, [ids[i : i + 150] for i in range(0, len(ids), 150)]):
            records.extend(batch)
    assert {r["CID"] for r in records} == set(ids)
    for name, value in (
        ("pubchem_records", records),
        ("pubchem_queries", queries),
        ("pubchem_membership", membership),
        ("download_manifest", manifest),
    ):
        with gzip.open(raw / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
    print(
        json.dumps(
            {"stage": "fetch_completed", "unique_CIDs": len(records), "queries": len(queries)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
