"""Extract review-only chain evidence; does not build a library or train a model."""

import argparse
from pathlib import Path

import yaml

from structure_pretrain.chain_extraction.pipeline import scan
from structure_pretrain.chain_extraction.reporting import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--version", choices=["2025", "2022"], default="2025")
    parser.add_argument("--output", type=Path, default=Path("data/private"))
    parser.add_argument("--config", type=Path, default=Path("configs/chain_extraction.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--start-block", type=int, default=0, help="Zero-based; later starts mark scan partial"
    )
    parser.add_argument(
        "--max-records", type=int, help="Maximum additional records this invocation"
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.start_block < 0 or (args.max_records is not None and args.max_records < 1):
        parser.error("start-block must be nonnegative and max-records positive")
    if not args.report_only:
        if not args.source:
            parser.error("--source is required for scanning")
        scan(
            args.source,
            args.version,
            args.output,
            config,
            args.resume,
            args.start_block,
            args.max_records,
        )
    report(args.output, config)


if __name__ == "__main__":
    main()
