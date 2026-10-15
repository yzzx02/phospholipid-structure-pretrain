"""Harvest review candidates; never creates an approved table or complete lipids."""

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_pretrain.pubchem_polar.cache import ChemistryCache, seed_from_review  # noqa: E402
from structure_pretrain.pubchem_polar.client import atomic_json  # noqa: E402
from structure_pretrain.pubchem_polar.freeze_chains import freeze_chains  # noqa: E402
from structure_pretrain.pubchem_polar.pipeline import harvest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pubchem_polar_queries.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/private/pubchem_polar_review"))
    parser.add_argument(
        "--chain-review", type=Path, default=Path("data/private/msdial_chain_review")
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache")))
        / "pubchem-polar-v01",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["stage1", "stage2", "full"],
        default=["stage1", "stage2", "full"],
    )
    parser.add_argument("--skip-chain-freeze", action="store_true")
    parser.add_argument(
        "--seed-review",
        type=Path,
        help="Reuse audited outputs with exactly matching chemistry fingerprint",
    )
    args = parser.parse_args()
    if not args.skip_chain_freeze:
        report = freeze_chains(
            args.chain_review, args.chain_review / "msdial_derived_chains_v0.1.tsv"
        )
        atomic_json(args.output / "chain_freeze_report.json", report)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if args.seed_review:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = ChemistryCache(args.cache_dir, config.get("limits", {}))
        try:
            print(
                f"Seeded {seed_from_review(cache, args.seed_review)} audited CID checkpoints",
                flush=True,
            )
        finally:
            cache.close()
    harvest(config, args.config, args.output, args.cache_dir, args.stages)


if __name__ == "__main__":
    main()
