"""Prepare source adapters from saved public downloads and private manual curation."""

import argparse
import csv
import io
import json
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from collect_reported_headgroups_v01 import ROOT, lmsd_sources, lsg_sources


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def prepare_lipidblast(raw):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(raw / "LipidBlast_Glycerolipids.zip") as outer:
        name = next(x for x in outer.namelist() if x.endswith(".xlsm"))
        data = outer.read(name)
    sources = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        strings = [
            "".join(s.itertext())
            for s in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)
        ]
        rels = {
            r.attrib["Id"]: r.attrib["Target"]
            for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        }
        book = ET.fromstring(z.read("xl/workbook.xml"))
        for sheet in book.find("m:sheets", ns):
            target = rels[
                sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
            ]
            # These are fixed-headgroup class sheets. Read three real stored rows per sheet,
            # not a new combinatorial expansion; retain every source file for later review.
            kept = 0
            with z.open(posixpath.normpath("xl/" + target)) as stream:
                for _, element in ET.iterparse(stream, events=("end",)):
                    if element.tag != "{" + ns["m"] + "}row":
                        continue
                    row = int(element.attrib["r"])
                    values = {}
                    for cell in element.findall("m:c", ns):
                        v = cell.find("m:v", ns)
                        if v is not None:
                            values[cell.attrib["r"].rstrip("0123456789")] = (
                                strings[int(v.text)] if cell.attrib.get("t") == "s" else v.text
                            )
                    element.clear()
                    smiles = values.get("F", "")
                    if row < 7 or not smiles or "P" not in smiles:
                        if row > 10:
                            break
                        continue
                    sources.append(
                        {
                            "source": "MS-DIAL",
                            "source_id_or_doi": f"LipidBlast_VS12:{sheet.attrib['name']}:row{row}",
                            "source_url": "https://zenodo.org/records/10953284",
                            "abbreviation": sheet.attrib["name"],
                            "full_name": values.get("E", sheet.attrib["name"]),
                            "aliases": [values["X"]] if values.get("X") else [],
                            "original_smiles": smiles,
                            "structure_input_kind": "full_lipid",
                            "source_kind": "theoretical_library_structure_literature_unverified",
                            "msms_status": "theoretical",
                            "msms_reference": (
                                "Official LipidBlast VS12 workbook stored fragmentation rules, "
                                "not experimental spectra"
                            ),
                            "workbook_source": name,
                            "source_sheet": sheet.attrib["name"],
                            "source_cell": f"F{row}",
                        }
                    )
                    kept += 1
                    if kept == 3:
                        break
    write_jsonl(raw / "msdial_lipidblast_structures.jsonl", sources)
    return len(sources)


def prepare_pubchem(raw):
    sources = []
    for row in json.loads((raw / "pubchem_cid_properties.json").read_text()):
        sources.append(
            {
                "source": "PubChem",
                "source_id_or_doi": f"CID:{row['CID']}",
                "source_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{row['CID']}",
                "pubchem_cid": str(row["CID"]),
                "abbreviation": "",
                "full_name": row.get("IUPACName", f"CID {row['CID']}"),
                "aliases": [],
                "original_smiles": row.get("SMILES", row.get("ConnectivitySMILES", "")),
                "structure_input_kind": "full_lipid",
                "source_kind": "database_only_literature_unverified",
                "msms_status": "unknown",
                "msms_reference": "",
            }
        )
    queries = json.loads((raw / "pubchem_name_queries.json").read_text())
    for query in queries:
        for row in query.get("records", []):
            sources.append(
                {
                    "source": "PubChem",
                    "source_id_or_doi": f"CID:{row['CID']}",
                    "source_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{row['CID']}",
                    "pubchem_cid": str(row["CID"]),
                    "abbreviation": "",
                    "full_name": row.get("IUPACName", ""),
                    "aliases": [query["query"]],
                    "original_smiles": row.get("SMILES", row.get("ConnectivitySMILES", "")),
                    "structure_input_kind": "full_lipid",
                    "source_kind": "database_only_literature_unverified",
                    "msms_status": "unknown",
                    "msms_reference": "",
                    "queried_name": query["query"],
                }
            )
    write_jsonl(raw / "pubchem_structures.jsonl", sources)
    return len(sources)


def prepare_literature(raw):
    rows = []
    path = raw / "literature_curation.csv"
    if not path.exists():
        return 0
    lsg = {r["abbreviation"]: r for r in lsg_sources(raw)}
    lmsd = {r["source_id_or_doi"]: r for r in lmsd_sources(raw)}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for curated in csv.DictReader(f):
            if curated.get("lsg_class"):
                base = dict(lsg[curated["lsg_class"]])
                base["structure_source_url"] = base["source_url"]
            elif curated.get("lm_id"):
                base = dict(lmsd[curated["lm_id"]])
                base["structure_source_url"] = base["source_url"]
            else:
                base = {
                    "original_smiles": curated["polar_smiles"],
                    "structure_input_kind": "standard_polar",
                    "original_smiles_provenance": (
                        "manual transcription of cited published structure; "
                        "[*:1] marks its drawn glycerol O-P boundary"
                    ),
                }
            base.update(
                {
                    "source": "Literature",
                    "source_id_or_doi": curated["doi"] or "PMID:" + curated["pmid"],
                    "source_url": curated["source_url"],
                    "abbreviation": curated["abbreviation"],
                    "full_name": curated["full_name"],
                    "aliases": [s.strip() for s in curated["aliases"].split(";") if s.strip()],
                    "source_kind": curated["source_kind"],
                    "literature_verified": True,
                    "evidence_locator": curated["evidence_locator"],
                    "msms_status": curated["msms_status"],
                    "msms_reference": curated["source_url"]
                    if curated["msms_status"] == "experimental"
                    else "",
                    "literature_link_scope": (
                        "headgroup structure/class, not every database parent acyl composition"
                    ),
                }
            )
            rows.append(base)
    write_jsonl(raw / "literature_structures.jsonl", rows)
    return len(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    raw = args.root / "raw"
    print(
        json.dumps(
            {
                "MSDIAL_stored_representatives": prepare_lipidblast(raw),
                "PubChem_records": prepare_pubchem(raw),
                "literature_links": prepare_literature(raw),
            }
        )
    )
