"""Build representative structures from externally supplied fragment tables; never train."""

import argparse
import json

from structure_pretrain.library_builder.pipeline import build_library
from structure_pretrain.library_builder.schemas import load_build_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/library_build.yaml")
    parser.add_argument("--chains", required=True)
    parser.add_argument("--polar", required=True)
    parser.add_argument("--p-to-o-mapping")
    parser.add_argument("--output", default="data/structure_library_v0.1")
    args = parser.parse_args()
    report = build_library(
        load_build_config(args.config), args.chains, args.polar, args.output, args.p_to_o_mapping
    )
    print(
        json.dumps(
            {
                field: report[field]
                for field in (
                    "status",
                    "data_kind",
                    "total_structures",
                    "number_of_chains",
                    "number_of_polar_moieties",
                    "AA_count",
                    "OA_count",
                    "PE_PA_count",
                    "lyso_count",
                    "rejected_count",
                    "training_performed",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
