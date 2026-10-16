"""Filter the existing APPROVABLE snapshot offline; never assemble or train."""

import argparse
import json

from structure_pretrain.polar_plausibility.pipeline import filter_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", default="data/private/pubchem_polar_review")
    parser.add_argument("--output", default="data/private/pubchem_polar_plausibility_v0.1")
    parser.add_argument("--config", default="configs/polar_plausibility.yaml")
    parser.add_argument("--rules", default="configs/polar_reactive_rules.yaml")
    parser.add_argument("--controls", default="configs/known_head_controls.tsv")
    args = parser.parse_args()
    report = filter_snapshot(args.review_dir, args.output, args.config, args.rules, args.controls)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
