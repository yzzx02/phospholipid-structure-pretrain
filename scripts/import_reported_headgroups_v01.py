"""Append user CSV/JSONL into a new, reproducible collection without changing the old library."""

import argparse
import csv
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

from collect_reported_headgroups_v01 import ROOT, collect, read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--library-root", type=Path, default=ROOT)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    if args.output_root.resolve().is_relative_to((args.library_root / "raw").resolve()):
        raise ValueError(
            "Output must be outside the source raw tree; recursive copying is forbidden"
        )
    if args.output_root.exists():
        raise ValueError("Choose a new output directory; existing libraries are never overwritten")
    if args.input.suffix.lower() == ".csv":
        with args.input.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = read_jsonl(args.input)
    required = {
        "full_name",
        "original_smiles",
        "source",
        "source_id_or_doi",
        "source_url",
        "structure_input_kind",
        "source_kind",
    }
    for row in rows:
        if not all(row.get(k) for k in required):
            raise ValueError(f"Required nonempty fields: {sorted(required)}")
        if row["structure_input_kind"] not in ("full_lipid", "standard_polar"):
            raise ValueError(
                "User imports require a full source lipid or an explicitly mapped standard Polar"
            )
        url = urlparse(row["source_url"])
        if url.scheme not in ("http", "https") or not url.netloc:
            raise ValueError("source_url must be an absolute HTTP(S) source link")
        row.setdefault("aliases", [])
        if isinstance(row["aliases"], str):
            if row["aliases"].lstrip().startswith("["):
                row["aliases"] = json.loads(row["aliases"])
            else:
                row["aliases"] = [s.strip() for s in row["aliases"].split(";") if s.strip()]
        if not isinstance(row["aliases"], list) or not all(
            isinstance(name, str) for name in row["aliases"]
        ):
            raise ValueError("aliases must be a list of names or a semicolon-separated string")
        row["literature_verified"] = row.get("literature_verified") in (True, "true", "True")
        row.setdefault("msms_status", "unknown")
    args.output_root.mkdir(parents=True)
    shutil.copytree(args.library_root / "raw", args.output_root / "raw")
    path = args.output_root / "raw/user_imports.jsonl"
    previous = read_jsonl(path)
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in previous + rows), encoding="utf-8"
    )
    collect(args.output_root)


if __name__ == "__main__":
    main()
