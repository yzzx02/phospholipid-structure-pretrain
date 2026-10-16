"""Simulate family-aware head/chain schedules; never assemble or train full lipids."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from structure_pretrain.library_builder.family_sampling import dry_run_family_schedule
from structure_pretrain.library_builder.full_library_v02 import load_config_v02, preflight_v02
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_plausibility.pipeline import dump_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/library_build_expanded_v02.yaml")
    parser.add_argument(
        "--chains", default="data/private/msdial_chain_review/msdial_derived_chains_v0.1.tsv"
    )
    parser.add_argument(
        "--polar", default="data/private/polar_expansion_v0.2/polar_moieties_expanded_v0.2.tsv"
    )
    parser.add_argument(
        "--report", default="data/private/polar_expansion_v0.2/polar_expansion_report.json"
    )
    parser.add_argument("--special-dir", default="data/private/special_cases_v0.2")
    parser.add_argument("--output", default="data/private/polar_expansion_v0.2/family_dry_run")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1500000, 2000000])
    args = parser.parse_args()
    config, options = load_config_v02(args.config)
    if options.family_sampling_weight is None:
        raise ValueError("Family-aware configuration required")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve existing dry-runs; choose an empty output directory")
    config = replace(config, max_structures=max(args.budgets))
    preflight, (_, polars, details, selected, bank) = preflight_v02(
        config, options, args.chains, args.polar, args.report, args.special_dir
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": Path(args.config),
        "chains": Path(args.chains),
        "polar": Path(args.polar),
        "expansion_report": Path(args.report),
    }
    paths.update({f"special:{p.name}": p for p in Path(args.special_dir).glob("*.tsv")})
    dump_json(
        output / "inputs_manifest.json",
        {
            key: {"path": str(path.resolve()), "sha256": file_hash(path)}
            for key, path in paths.items()
        },
    )
    dump_json(output / "preflight.json", preflight)
    summaries = []
    for maximum in args.budgets:
        current = replace(config, max_structures=maximum)
        summary, plan, counts = dry_run_family_schedule(
            polars, details, selected, bank, current, options.family_sampling_weight
        )
        summary["common_anchor_ids"] = preflight["common_anchor_ids"]
        summary["phosphate_full_exceeds_phosphonate"] = summary[
            "estimated_full_by_chemistry_family"
        ].get("phosphate", 0) > summary["estimated_full_by_chemistry_family"].get("phosphonate", 0)
        summary["sampling_weight_adjustment_needed"] = not summary[
            "phosphate_full_exceeds_phosphonate"
        ]
        dump_json(output / f"dry_run_{maximum}.json", summary)
        rows = [
            {
                "polar_id": p.polar_id,
                "chemistry_family": details[p.polar_id]["chemistry_family"],
                "structure_origin": details[p.polar_id]["structure_origin"],
                "mandatory_count": len(selected[p.polar_id]),
                "extra_quota": plan.extra_quotas[p.polar_id],
                "lyso_quota": plan.lyso_quotas[p.polar_id],
                "planned_background_count": counts[p.polar_id],
                "is_special_case": details[p.polar_id]["is_special_case"],
            }
            for p in polars
        ]
        pq.write_table(
            pa.Table.from_pylist(rows),
            output / f"per_head_budget_{maximum}.parquet",
            compression="zstd",
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    dump_json(output / "budget_comparison.json", summaries)


if __name__ == "__main__":
    main()
