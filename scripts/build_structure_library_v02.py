"""Default: validate inputs and estimate budget only. --execute explicitly starts full assembly."""

import argparse
import json

from structure_pretrain.library_builder.full_library_v02 import (
    build_v02,
    load_config_v02,
    preflight_v02,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/library_build_v02.yaml")
    parser.add_argument(
        "--chains", default="data/private/msdial_chain_review/msdial_derived_chains_v0.1.tsv"
    )
    parser.add_argument(
        "--polar",
        default="data/private/pubchem_polar_plausibility_v0.1/polar_moieties_pubchem_v0.1.tsv",
    )
    parser.add_argument(
        "--plausibility-report",
        default="data/private/pubchem_polar_plausibility_v0.1/polar_plausibility_report.json",
    )
    parser.add_argument("--special-dir", default="data/private/special_cases_v0.2")
    parser.add_argument("--output", default="data/structure_library_v0.2")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config, options = load_config_v02(args.config)
    if args.execute:
        report = build_v02(
            config,
            options,
            args.chains,
            args.polar,
            args.plausibility_report,
            args.output,
            args.special_dir,
            args.scratch_dir,
        )
    else:
        report, _ = preflight_v02(
            config, options, args.chains, args.polar, args.plausibility_report, args.special_dir
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
