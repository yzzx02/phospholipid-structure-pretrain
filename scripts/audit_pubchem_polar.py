"""Cross-check persisted review tables and provenance, without network access."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem


def rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
        yield from batch.to_pylist()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory / "pubchem_polar_report.json").read_text(encoding="utf-8"))
    raw_ids, query_counts = set(), Counter()
    for row in rows(directory / "pubchem_polar_raw.parquet"):
        require(row["cid"] not in raw_ids, "Duplicate raw CID")
        raw_ids.add(row["cid"])
        query_counts.update(row["matched_query_ids"])
    require(len(raw_ids) == report["properties_downloaded"], "Raw row count differs from report")
    require(
        len(raw_ids) + len(report["missing_property_cids"]) == report["unique_cids"],
        "Query/property accounting mismatch",
    )
    if report["properties_complete"]:
        require(
            query_counts
            == Counter({r["query_id"]: r["cid_count"] for r in report["query_results"]}),
            "Raw query provenance differs from query summary",
        )
    component_ids = set()
    for row in rows(directory / "pubchem_p_components.parquet"):
        require(row["cid"] in raw_ids and row["cid"] not in component_ids, "Bad component CID")
        require("." not in row["selected_p_component"], "Selected component still disconnected")
        component_ids.add(row["cid"])
    reject_ids, rejection_counts = set(), Counter()
    for row in rows(directory / "pubchem_polar_rejections.parquet"):
        require(row["cid"] in raw_ids and row["cid"] not in reject_ids, "Bad rejection CID")
        reject_ids.add(row["cid"])
        rejection_counts[row["reason"]] += 1
    require(rejection_counts == Counter(report["rejections"]), "Rejection accounting mismatch")
    expected_sites, expected_sources = {}, {}
    family_counts, status_counts = Counter(), Counter()
    fragment_count = 0
    for row in rows(directory / "pubchem_polar_fragments.parquet"):
        key, polar_id = row["polar_fragment_smiles"], row["polar_id"]
        require(polar_id == "PCP_" + hashlib.sha256(key.encode()).hexdigest()[:16], "Bad polar ID")
        require(polar_id not in expected_sites, "Duplicate fragment or hash collision")
        require(len(set(row["source_cids"])) == row["source_count"], "Duplicate source provenance")
        require(set(row["source_cids"]) <= raw_ids, "Fragment references unknown raw CID")
        expected_sites[polar_id] = row["attachment_site_count_before_dedup"]
        expected_sources[polar_id] = row["source_count"]
        mol = Chem.MolFromSmiles(key)
        require(mol is not None, "Invalid fragment SMILES")
        dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        require(
            len(dummies) == 1
            and dummies[0].GetAtomMapNum() == 1
            and dummies[0].GetDegree() == 1
            and dummies[0].GetNeighbors()[0].GetAtomicNum() == 15,
            "Bad P attachment boundary",
        )
        require(Chem.GetFormalCharge(mol) == row["formal_charge"], "Charge descriptor mismatch")
        require(not row["allow_pa"] or row["known_control_ids"] == ["PE"], "Non-PE P/A permission")
        if row["status"] == "APPROVABLE":
            require(
                row["formal_charge"] == 0
                and not row["filter_flags"]
                and row["review_flags"] == ["APPROVABLE"],
                "APPROVABLE has unresolved flags",
            )
        family_counts[row["polar_family"]] += 1
        status_counts[row["status"]] += 1
        fragment_count += 1
    require(fragment_count == report["unique_fragments"], "Fragment count mismatch")
    require(family_counts == Counter(report["family_counts"]), "Family count mismatch")
    require(status_counts == Counter(report["status_counts"]), "Status count mismatch")
    candidate_ids, current_keys, observed_sites, observed_sources = (
        set(),
        set(),
        Counter(),
        Counter(),
    )
    current_cid = None
    candidate_count = 0
    for row in rows(directory / "pubchem_attachment_candidates.parquet"):
        cid = row["cid"]
        require(cid in component_ids, "Candidate has no selected component")
        if cid != current_cid:
            observed_sources.update(current_keys)
            require(cid not in candidate_ids, "Candidate CIDs not contiguous")
            current_cid, current_keys = cid, set()
            candidate_ids.add(cid)
        polar_id = "PCP_" + hashlib.sha256(row["polar_fragment_smiles"].encode()).hexdigest()[:16]
        require(polar_id in expected_sites, "Candidate missing from fragment table")
        observed_sites[polar_id] += 1
        current_keys.add(polar_id)
        candidate_count += 1
    observed_sources.update(current_keys)
    require(observed_sites == Counter(expected_sites), "Attachment aggregation mismatch")
    require(observed_sources == Counter(expected_sources), "Source-CID aggregation mismatch")
    require(
        not candidate_ids & reject_ids and candidate_ids | reject_ids == raw_ids,
        "Raw records are not partitioned into attachment-eligible/rejected",
    )
    require(candidate_count == report["attachment_candidates"], "Candidate count mismatch")
    require(not (directory / "polar_moieties_pubchem_v0.1.tsv").exists(), "Approved table present")
    result = {
        "passed": True,
        "raw_cids": len(raw_ids),
        "selected_components": len(component_ids),
        "rejected_cids": len(reject_ids),
        "attachment_eligible_cids": len(candidate_ids),
        "attachment_candidates": candidate_count,
        "unique_fragments": fragment_count,
        "source_cid_fragment_pairs": sum(observed_sources.values()),
        "fragment_graphs_checked": fragment_count,
        "query_provenance_checked": True,
        "approved_table_absent": True,
    }
    (directory / "verification_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="data/private/pubchem_polar_review")
    print(json.dumps(audit(parser.parse_args().directory), indent=2))


if __name__ == "__main__":
    main()
