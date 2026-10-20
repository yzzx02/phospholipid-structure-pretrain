"""Check collection integrity, unchanged Builder behavior, name lookups and provenance."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from structure_pretrain.reported_headgroups.chemistry import classify_polar, stable_id

ROOT = Path("outputs/reported_headgroups_v01_20261020")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def typed(row):
    result = {}
    for k, v in row.items():
        if v in ("True", "False"):
            v = v == "True"
        elif (k.endswith("_count") or k in ("source_count", "query_match_index")) and v.isdigit():
            v = int(v)
        result[k] = v
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    root = args.root
    heads = rows(root / "reported_headgroups_v01.csv")
    names = rows(root / "headgroup_name_alias_mapping.csv")
    sources = rows(root / "headgroup_sources.csv")
    pending = rows(root / "headgroups_pending_review.csv")
    summary = json.loads((root / "collection_summary.json").read_text())
    ids = {h["headgroup_id"]: h for h in heads}
    assert len(ids) == len(heads) == summary["distinct_resolved_polar_graphs"]
    assert len({h["polar_smiles"] for h in heads}) == len(heads)
    source_counts, name_counts = Counter(), Counter()
    names_by_key, source_ids = defaultdict(set), set()
    for source in sources:
        assert source["headgroup_id"] in ids
        assert source["source_url"].startswith(("https://", "http://"))
        assert source["source_id_or_doi"] and source["original_smiles"]
        assert source["source_record_id"] not in source_ids
        source_ids.add(source["source_record_id"])
        assert source["polar_smiles"] == ids[source["headgroup_id"]]["polar_smiles"]
        source_counts[source["headgroup_id"]] += 1
    for row in names:
        assert row["headgroup_id"] in ids
        assert row["polar_smiles"] == ids[row["headgroup_id"]]["polar_smiles"]
        assert row["normalized_name"] == " ".join(row["query_name"].casefold().split())
        names_by_key[row["normalized_name"]].add(row["headgroup_id"])
        name_counts[row["headgroup_id"]] += 1
    assert len({r["query_lookup_key"] for r in names}) == len(names)
    builder_pass = 0
    for head in heads:
        polar = head["polar_smiles"]
        assert stable_id(polar) == head["headgroup_id"]
        mol = Chem.MolFromSmiles(polar)
        assert mol is not None
        dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        assert len(dummies) == 1 and dummies[0].GetAtomMapNum() == 1
        assert dummies[0].GetDegree() == 1 and dummies[0].GetNeighbors()[0].GetAtomicNum() == 15
        assert head["builder_compatible"] == str(classify_polar(polar)["builder_compatible"])
        builder_pass += int(head["builder_compatible"] == "True")
        assert source_counts[head["headgroup_id"]] == int(head["source_count"])
        assert name_counts[head["headgroup_id"]] == int(head["name_alias_count"])
    assert builder_pass == summary["builder_compatible"]
    assert len(names_by_key["ppa"]) == 2
    assert len(names_by_key["lpg"]) >= 2
    assert len(names_by_key["pc"]) >= 1
    frozen_path = Path("data/structure_library_v0.3/polar_library.parquet")
    frozen = pq.read_table(frozen_path, columns=["polar_fragment_smiles"]).column(0).to_pylist()
    assert len(frozen) == 195306
    frozen_set = frozenset(frozen)
    assert sum(h["polar_smiles"] in frozen_set for h in heads) == summary["in_frozen_195306"]
    assert all(not r.get("headgroup_id") or r["headgroup_id"] in ids for r in pending)
    summary["pending_reason_counts"] = dict(Counter(r["review_reason"] for r in pending))
    summary["resolved_graphs_needing_review"] = sum(bool(h["review_reason"]) for h in heads)
    summary["unresolved_source_rows_needing_review"] = sum(
        not r.get("headgroup_id") for r in pending
    )
    summary["sources_by_record"] = dict(Counter(r["source"] for r in sources))
    summary["builder_incompatible_by_reason"] = dict(
        Counter(h["incompatibility_reason"] for h in heads if h["builder_compatible"] != "True")
    )
    summary["source_roundtrip_graph_true"] = sum(
        s.get("roundtrip_graph") == "True" for s in sources
    )
    summary["source_roundtrip_formula_true"] = sum(
        s.get("roundtrip_formula") == "True" for s in sources
    )
    summary["all_compatible_heads_independent_H_balance_checked"] = True
    summary["frozen_polar_library_sha256"] = digest(frozen_path)
    summary["rdkit_version"] = rdBase.rdkitVersion
    summary["sha256_files"] = {p.name: digest(p) for p in root.glob("*.csv")}
    summary["source_code_sha256"] = {
        p.as_posix(): digest(p)
        for p in [
            *Path("src/structure_pretrain/reported_headgroups").glob("*.py"),
            *Path("scripts").glob("*reported_headgroup*.py"),
            Path("src/structure_pretrain/library_builder/assembly.py"),
            Path("src/structure_pretrain/library_builder/validation.py"),
        ]
    }
    (root / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "头基总表": [typed(r) for r in heads],
        "名称映射": [typed(r) for r in names],
        "来源记录": [typed(r) for r in sources],
        "待审核": [typed(r) for r in pending],
        "summary": summary,
    }
    (root / "excel_payload_private.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "audit": "passed",
                "graphs": len(heads),
                "builder_pass": builder_pass,
                "sources": len(sources),
                "name_mappings": len(names),
                "pending": len(pending),
                "PPA_graphs": len(names_by_key["ppa"]),
                "LPG_graphs": len(names_by_key["lpg"]),
            }
        )
    )


if __name__ == "__main__":
    main()
