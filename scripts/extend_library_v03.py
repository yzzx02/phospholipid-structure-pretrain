"""Append approximately 15,000 unused validated head/chain combinations to frozen v0.2."""

import argparse
import json

from structure_pretrain.library_builder.incremental import build_increment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="data/structure_library_v0.2")
    parser.add_argument("--cache", default="data/private/morgan_readiness_v0.1")
    parser.add_argument("--output", default="data/structure_library_v0.3")
    parser.add_argument(
        "--baseline-db", default="data/private/library_2m_assembly_work/canonical_seen.sqlite"
    )
    parser.add_argument(
        "--chains", default="data/private/msdial_chain_review/msdial_derived_chains_v0.1.tsv"
    )
    parser.add_argument(
        "--config", default="data/private/library_2m_preparation_v0.1/library_build_2m_final.yaml"
    )
    parser.add_argument("--planned", type=int, default=15000)
    args = parser.parse_args()
    if not 10000 <= args.planned <= 20000:
        raise ValueError("Increment proposals must be between 10,000 and 20,000; no success refill")
    print(
        json.dumps(
            build_increment(
                args.library,
                args.cache,
                args.output,
                args.baseline_db,
                args.chains,
                args.config,
                args.planned,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
