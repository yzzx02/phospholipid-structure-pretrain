"""Read-only name/alias or core-ID lookup; returns every ambiguous name match."""

import argparse
import gzip
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--root", default="outputs/glycerol_core_collection_v01_20261023")
    parser.add_argument("--source-limit", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.root)
    aliases = pq.read_table(root / "core_name_alias_index.parquet")
    mask = pc.match_substring(pc.utf8_lower(aliases["name_or_alias"]), args.query.lower())
    ids = set(aliases.filter(mask)["core_id"].to_pylist())
    ids.add(args.query.upper())
    rows = [
        r
        for r in json.loads((root / "core_catalog_private.json").read_text(encoding="utf-8"))
        if r["core_id"] in ids
    ]
    by_id = {r["core_id"]: r for r in rows}
    for row in rows:
        row["representative_source_records"] = []
    with gzip.open(root / "core_sources.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            if source["core_id"] in by_id:
                row = by_id[source["core_id"]]
                if len(row["representative_source_records"]) < args.source_limit:
                    row["representative_source_records"].append(source)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
