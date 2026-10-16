"""Offline polar expansion only. Never train or execute full lipid assembly."""

import argparse
import json

from structure_pretrain.polar_expansion.pipeline import expand_polars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", default="data/private/pubchem_polar_review")
    parser.add_argument(
        "--baseline",
        default="data/private/pubchem_polar_plausibility_v0.1/polar_moieties_pubchem_v0.1.tsv",
    )
    parser.add_argument("--output", default="data/private/polar_expansion_v0.2")
    parser.add_argument("--config", default="configs/polar_expansion.yaml")
    parser.add_argument("--plausibility", default="configs/polar_plausibility.yaml")
    parser.add_argument("--rules", default="configs/polar_reactive_rules.yaml")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    report = expand_polars(
        args.review_dir,
        args.baseline,
        args.output,
        args.config,
        args.plausibility,
        args.rules,
        args.workers,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
