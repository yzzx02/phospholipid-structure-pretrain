"""Apply explicit human review and collect source-grounded v0.2 additions on G drive."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from collect_reported_headgroups_v01 import collect, read_csv, read_jsonl, write_csv
from rdkit import Chem


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_approved(previous, review_json, review_csv, output):
    heads = read_csv(previous / "reported_headgroups_v01.csv")
    by_id = {r["headgroup_id"]: r for r in heads}
    review = json.loads(review_json.read_text(encoding="utf-8-sig"))
    assert review["schema"] == "headgroup-review-v01"
    assert review["input_sha256"] == digest(previous / "reported_headgroups_v01.csv")
    csv_review = read_csv(review_csv)
    assert len(csv_review) == len(by_id)
    assert {r["headgroup_id"] for r in csv_review} == by_id.keys()
    decisions = review["decisions"]
    assert all(
        d.get("decision", "") in {"", "keep", "exclude", "review"} for d in decisions.values()
    )
    for row in csv_review:
        identifier = row["headgroup_id"]
        assert row["polar_smiles"] == by_id[identifier]["polar_smiles"]
        assert row["human_decision"] == decisions.get(identifier, {}).get("decision", "")
        assert row["human_note"] == decisions.get(identifier, {}).get("note", "")
    approved = {k for k, v in decisions.items() if v.get("decision") == "keep"}
    assert set(decisions) <= by_id.keys()
    selected = [
        {**r, "human_decision": "keep", "human_note": decisions[r["headgroup_id"]].get("note", "")}
        for r in heads
        if r["headgroup_id"] in approved
    ]
    directory = output / "approved_v01"
    directory.mkdir(exist_ok=True)
    for src in (review_json, review_csv):
        dest = directory / src.name
        if dest.exists():
            assert digest(src) == digest(dest), "Never replace a different human review snapshot"
        else:
            shutil.copyfile(src, dest)
    write_csv(directory / "approved_headgroups_v01.csv", selected)
    for filename in ("headgroup_sources.csv", "headgroup_name_alias_mapping.csv"):
        write_csv(
            directory / filename,
            [r for r in read_csv(previous / filename) if r["headgroup_id"] in approved],
        )
    receipt = {
        "decision_counts": dict(Counter(d["decision"] for d in decisions.values())),
        "approved_graphs": len(selected),
        "approved_builder_pass": sum(r["builder_compatible"] == "True" for r in selected),
        "source_sha256": review["input_sha256"],
        "review_json_sha256": digest(review_json),
        "review_csv_sha256": digest(review_csv),
        "review_exported_at": review.get("exported_at"),
        "training_started": False,
    }
    (directory / "approval_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous", type=Path, default=Path("outputs/reported_headgroups_v01_20261020")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reported_headgroups_v02_20261021")
    )
    parser.add_argument(
        "--review-json", type=Path, default=Path("G:/yzx/linzhiyuce/headgroup_manual_review.json")
    )
    parser.add_argument(
        "--review-csv", type=Path, default=Path("G:/yzx/linzhiyuce/headgroup_manual_review.csv")
    )
    parser.add_argument("--approval-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = save_approved(args.previous, args.review_json, args.review_csv, args.output)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if args.approval_only:
        return
    raw = args.output / "raw"
    records = json.loads((raw / "pubchem_records.json").read_text(encoding="utf-8"))
    queries = json.loads((raw / "pubchem_cid_queries.json").read_text(encoding="utf-8"))
    synonym_file = raw / "pubchem_synonyms.json"
    synonyms = (
        {
            r["CID"]: r.get("Synonym", [])
            for r in json.loads(synonym_file.read_text(encoding="utf-8"))
        }
        if synonym_file.exists()
        else {}
    )
    sources, outcomes = [], Counter()
    for row in records:
        smiles = row.get("SMILES", row.get("ConnectivitySMILES", ""))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not any(a.GetAtomicNum() == 15 for a in mol.GetAtoms()):
            outcomes["no_parseable_phosphorus_graph"] += 1
            continue
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("[O]-[CH2]-[CH](-[O])-[CH2]-[O]-[P]")):
            outcomes["no_glycerol_O_P_subgraph"] += 1
            continue
        cid = row["CID"]
        sources.append(
            {
                "source": "PubChem",
                "source_id_or_doi": f"CID:{cid}",
                "source_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                "pubchem_cid": str(cid),
                "full_name": row.get("IUPACName", f"CID {cid}"),
                "abbreviation": "",
                "aliases": synonyms.get(cid, []),
                "original_smiles": smiles,
                "structure_input_kind": "full_lipid",
                "source_kind": "database_only_literature_unverified",
                "msms_status": "unknown",
                "queried_names_or_terms": queries[str(cid)],
                "evidence_locator": (
                    "Official PUG REST structure properties; actual glycerol O-P graph required"
                ),
            }
        )
        outcomes["glycerol_O_P_source_records"] += 1
    sources += read_jsonl(raw / "curated_additions.jsonl")
    path = raw / "v02_additional_sources.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sources), encoding="utf-8"
    )
    before = digest(args.previous / "reported_headgroups_v01.csv")
    collect(args.output, extra=path, source_raw=args.previous / "raw")
    assert before == digest(args.previous / "reported_headgroups_v01.csv")
    old = {r["headgroup_id"] for r in read_csv(args.previous / "reported_headgroups_v01.csv")}
    all_heads = read_csv(args.output / "reported_headgroups_v01.csv")
    write_csv(args.output / "reported_headgroups_v02.csv", all_heads)
    write_csv(
        args.output / "new_headgroup_graphs.csv",
        [r for r in all_heads if r["headgroup_id"] not in old],
    )
    summary = {
        "approval": receipt,
        "pubchem_search_outcomes": dict(outcomes),
        "new_distinct_graphs": sum(r["headgroup_id"] not in old for r in all_heads),
        "total_distinct_graphs_in_raw_reference": len(all_heads),
        "previous_graphs_all_preserved": old <= {r["headgroup_id"] for r in all_heads},
    }
    (args.output / "expansion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
