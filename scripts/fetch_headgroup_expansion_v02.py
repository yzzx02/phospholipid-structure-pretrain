"""Cache bounded PubChem name/text searches and primary literature for headgroup curation."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

import requests

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
NAMES = [
    "Amadori phosphatidylethanolamine",
    "Amadori phosphatidylserine",
    "phosphatidylglucose",
    "phosphatidylglucoside",
    "phosphatidylgalactose",
    "phosphatidylmannose",
    "phosphatidylfructose",
    "phosphatidylribose",
    "lysylphosphatidylglycerol",
    "alanylphosphatidylglycerol",
    "arginylphosphatidylglycerol",
    "ornithylphosphatidylglycerol",
    "glycylphosphatidylglycerol",
    "phosphatidylthreonine",
    "phosphatidylhomoserine",
    "phosphatidylserinol",
    "phosphatidyldiethanolamine",
    "phosphatidyltriethanolamine",
    "phosphatidyltris",
    "phosphatidylglyceric acid",
    "phosphatidylglycol",
    "phosphatidylpropanolamine",
    "phosphatidyldopamine",
    "phosphatidyltyrosine",
    "phosphatidylglycine",
    "phosphatidylalanine",
    "phosphatidyltryptophan",
    "phosphatidylhydroxyproline",
    "phosphatidylhydroxyethylglycine",
    "N-carboxyethylphosphatidylethanolamine",
    "N-carboxymethylphosphatidylserine",
    "N-acetylphosphatidylethanolamine",
    "N-formylphosphatidylethanolamine",
    "phosphatidylhydroxyethanol",
    "phosphatidylpropanol",
    "phosphatidylbutanol",
]
TEXT_TERMS = {
    "amadori": (
        "(amadori OR glycated OR fructosyl OR carboxyethyl OR carboxymethyl) "
        "AND (phosphatidyl OR phosphoethanolamine OR phosphoserine)"
    ),
    "amino_heads": (
        "(phosphatidyl OR phosphatidylglycerol) AND (arginyl OR ornithyl OR glycyl OR alanyl "
        "OR lysyl OR aminoacyl OR homoserine OR serinol OR threonine "
        "OR diethanolamine OR triethanolamine)"
    ),
    "nonclassical_gp": (
        '"glycero-3-phospho"[Synonym] NOT '
        "(choline OR ethanolamine OR inositol OR serine OR glycerol)"
    ),
    "phosphatidyl_other": (
        "phosphatidyl[Synonym] NOT (phosphatidylcholine OR phosphatidylethanolamine OR "
        "phosphatidylinositol OR phosphatidylserine OR phosphatidylglycerol OR cardiolipin)"
    ),
    "phosphatidyl_all": "phosphatidyl*[Synonym]",
    "glycero_phospho_all": '"glycero" AND "phospho"',
    "phosphoester_names": "(phosphoryl OR phosphate) AND (glyceryl OR glycerol OR glycerophospho)",
}
PAPERS = {
    "amadori": "PMC6275948",
    "aminoacyl_PG": "PMC4553816",
    "aminoacyl_review": "PMC2291102",
    "glyco_GP_review": "PMC10881803",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-limit", type=int, default=4000)
    args = parser.parse_args()
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = []

    def cached(label, url, suffix="json"):
        file = raw / (label + "." + suffix)
        status = None
        if not file.exists():
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=90)
                    status = response.status_code
                    if status in (429, 500, 502, 503, 504):
                        time.sleep(1 + attempt)
                        continue
                    file.write_bytes(response.content)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
            else:
                file.write_bytes(response.content)
            time.sleep(0.55)
        entry = {
            "file": str(file.relative_to(args.output)),
            "url": url,
            "http_status_on_fetch": status,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
        manifest.append(entry)
        if suffix == "json":
            try:
                return json.loads(file.read_bytes())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {"download_error": file.name}
        return None

    queries, cid_queries = [], {}
    for name in NAMES:
        label = "name_" + hashlib.sha256(name.encode()).hexdigest()[:12]
        data = cached(label, PUG + "/compound/name/" + quote(name, safe="") + "/cids/JSON")
        ids = data.get("IdentifierList", {}).get("CID", [])
        if not ids and data.get("Fault", {}).get("Code") != "PUGREST.NotFound":
            raise ValueError(f"Unexpected PubChem name response, not a negative result: {label}")
        queries.append({"kind": "name", "query": name, "cids": ids, "result_count": len(ids)})
        for cid in ids:
            cid_queries.setdefault(int(cid), []).append(name)
    for key, term in TEXT_TERMS.items():
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(
            {"db": "pccompound", "term": term, "retmax": args.text_limit, "retmode": "json"}
        )
        data = cached("text_" + key, url)
        result = data.get("esearchresult", {})
        if not result:
            raise ValueError(f"Missing Entrez search response: {key}")
        ids = result.get("idlist", [])
        queries.append(
            {
                "kind": "Entrez",
                "query": term,
                "label": key,
                "reported_total": result.get("count"),
                "downloaded": len(ids),
                "cids": ids,
            }
        )
        for cid in ids:
            cid_queries.setdefault(int(cid), []).append(key)
    cids = sorted(cid_queries)
    print(json.dumps({"unique_queried_cids": len(cids), "searches": len(queries)}), flush=True)
    batches = [cids[i : i + 200] for i in range(0, len(cids), 200)]

    def properties(batch):
        token = hashlib.sha256(",".join(map(str, batch)).encode()).hexdigest()[:12]
        url = PUG + "/compound/cid/" + ",".join(map(str, batch))
        url += "/property/SMILES,ConnectivitySMILES,IUPACName,MolecularFormula/JSON"
        data = cached("properties_" + token, url)
        if "PropertyTable" not in data:
            raise ValueError(f"Missing PubChem properties: {token}")
        return data["PropertyTable"].get("Properties", [])

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = [record for batch in pool.map(properties, batches) for record in batch]
    if len(records) != len(cids) or {r["CID"] for r in records} != set(cids):
        raise ValueError("PubChem property coverage does not match the requested unique CIDs")
    (raw / "pubchem_records.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    (raw / "pubchem_queries.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (raw / "pubchem_cid_queries.json").write_text(
        json.dumps(cid_queries, ensure_ascii=False), encoding="utf-8"
    )
    for key, pmc in PAPERS.items():
        cached(
            "literature_" + key,
            "https://www.ebi.ac.uk/europepmc/webservices/rest/" + pmc + "/fullTextXML",
            "xml",
        )
    for name in ("phosphatidylserine", "phosphatidylethanolamine", "phosphatidylglucose"):
        cached(
            "nist_" + name,
            "https://webbook.nist.gov/cgi/cbook.cgi?" + urlencode({"Name": name, "Units": "SI"}),
            "html",
        )
    (raw / "download_manifest.json").write_text(
        json.dumps(
            {"downloaded_utc": datetime.now(timezone.utc).isoformat(), "files": manifest},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"property_records": len(records), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
