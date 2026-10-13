import argparse
import json
from pathlib import Path

from structure_pretrain.library_builder.msdial_inventory import inspect_lbm2


def main():
    parser = argparse.ArgumentParser(
        description="Inspect local LBM2 container counts and limited structure metadata"
    )
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--v5-metadata", action="store_true", help="Use verified v5 schema; do not use for v4 files"
    )
    parser.add_argument("--sample-records", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.sample_records <= 10000:
        parser.error("sample-records must be between 0 and 10000")
    result = inspect_lbm2(args.library, args.v5_metadata, args.sample_records)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "bytes": result["bytes"],
                "records": result["total_records_from_array_headers"],
                "blocks": len(result["blocks"]),
                "sampled": len(result["sample_records"]),
                "sample_classes": result["sample_compound_classes"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
