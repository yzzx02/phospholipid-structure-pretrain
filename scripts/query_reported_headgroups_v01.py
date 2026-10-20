"""Resolve every graph matching a name, or reverse-resolve all names and sources for an ID."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path("outputs/reported_headgroups_v01_20261020")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def query(root, name=None, identifier=None):
    names = rows(root / "headgroup_name_alias_mapping.csv")
    norm = " ".join(name.casefold().split()) if name else None
    ids = (
        {r["headgroup_id"] for r in names if r["normalized_name"] == norm} if norm else {identifier}
    )
    heads = [r for r in rows(root / "reported_headgroups_v01.csv") if r["headgroup_id"] in ids]
    sources = rows(root / "headgroup_sources.csv")
    return [
        {
            **h,
            "all_names": sorted(
                {r["query_name"] for r in names if r["headgroup_id"] == h["headgroup_id"]}
            ),
            "all_sources": [r for r in sources if r["headgroup_id"] == h["headgroup_id"]],
        }
        for h in heads
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--name")
    which.add_argument("--id", dest="identifier")
    p.add_argument("--compact", action="store_true")
    args = p.parse_args()
    result = query(args.root, args.name, args.identifier)
    if args.compact:
        result = [
            {
                k: r[k]
                for k in (
                    "headgroup_id",
                    "abbreviation",
                    "full_name",
                    "polar_smiles",
                    "source_count",
                )
            }
            for r in result
        ]
    print(
        json.dumps(
            {"matches": len(result), "ambiguous_name": len(result) > 1, "records": result},
            ensure_ascii=False,
            indent=2,
        )
    )
