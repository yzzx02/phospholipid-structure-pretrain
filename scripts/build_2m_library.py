"""Formally assemble the approved <=2M schedule, without training or a dataset split."""

import argparse
import json

from structure_pretrain.library_builder.production import build_production


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparation", default="data/private/library_2m_preparation_v0.1")
    p.add_argument(
        "--chains", default="data/private/msdial_chain_review/msdial_derived_chains_v0.1.tsv"
    )
    p.add_argument(
        "--polar", default="data/private/polar_expansion_v0.2/polar_moieties_expanded_v0.2.tsv"
    )
    p.add_argument(
        "--report", default="data/private/polar_expansion_v0.2/polar_expansion_report.json"
    )
    p.add_argument("--special-dir", default="data/private/special_cases_v0.2")
    p.add_argument("--output", default="data/structure_library_v0.2")
    p.add_argument("--work", default="data/private/library_2m_assembly_work")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--heads-per-job", type=int, default=800)
    p.add_argument("--execute", action="store_true", required=True)
    a = p.parse_args()
    if not 1 <= a.workers <= 4 or not 1 <= a.heads_per_job <= 2000:
        raise ValueError("Use 1-4 workers and 1-2000 heads per bounded job")
    report = build_production(
        a.preparation,
        a.chains,
        a.polar,
        a.report,
        a.special_dir,
        a.output,
        a.work,
        a.workers,
        a.heads_per_job,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
