"""Audit nitrile chemistry and assemble only a small keto-PI/nitrile representative set."""

import argparse
import json

from structure_pretrain.library_builder.special_cases_v02 import audit_special_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", default="data/private/pubchem_polar_review")
    parser.add_argument(
        "--chains", default="data/private/msdial_chain_review/msdial_derived_chains_v0.1.tsv"
    )
    parser.add_argument("--approved-polar")
    parser.add_argument("--output", default="data/private/special_cases_v0.2")
    args = parser.parse_args()
    report = audit_special_cases(args.review_dir, args.chains, args.output, args.approved_polar)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
