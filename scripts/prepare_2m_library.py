"""Audit the original 2M schedule; P exposure is descriptive, not a coverage gate."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from structure_pretrain.library_builder.coverage import chain_coverage
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
    parser.add_argument(
        "--reference-dry-run",
        default="data/private/polar_expansion_v0.2/family_dry_run/dry_run_2000000.json",
    )
    parser.add_argument("--output", default="data/private/library_2m_preparation_v0.1")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve existing preparation; choose an empty output directory")
    config, options = load_config_v02(args.config)
    if config.max_structures != 2_000_000 or options.family_sampling_weight != {
        "phosphate": 2.0,
        "phosphonate": 1.0,
        "polyphosphate": 1.0,
    }:
        raise ValueError("This preparation requires the approved 2M / 2:1:1 configuration")
    preflight, (chains, polars, details, selected, bank) = preflight_v02(
        config, options, args.chains, args.polar, args.report, args.special_dir
    )
    summary, _, _ = dry_run_family_schedule(
        polars, details, selected, bank, config, options.family_sampling_weight
    )
    reference = json.loads(Path(args.reference_dry_run).read_text(encoding="utf-8"))
    if summary["schedule_sha256"] != reference["schedule_sha256"]:
        raise ValueError(
            "The 2M main sampling schedule changed; no P coverage modifications allowed"
        )
    chain_report = chain_coverage(chains, summary["chain_token_counts"])
    if chain_report["status"] != "passed" or not summary["all_special_heads_retained"]:
        raise ValueError("A/O chain or special coverage failed; formal assembly blocked")
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "family_dry_run_v02.json", summary)
    dump_json(output / "chain_token_coverage_dry_run.json", chain_report)
    dump_json(
        output / "p_chain_coverage_report.json",
        {
            **chain_report["families"]["P"],
            "mandatory_pairs_added": 0,
            "policy": chain_report["p_policy"],
            "main_sampling_schedule_unchanged": True,
        },
    )
    resolved = {**asdict(config), **asdict(options)}
    (output / "library_build_2m_final.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    paths = {
        "base_config": Path(args.config),
        "chains": Path(args.chains),
        "polar": Path(args.polar),
        "report": Path(args.report),
        "reference_dry_run": Path(args.reference_dry_run),
        "final_config": output / "library_build_2m_final.yaml",
        "dry_run": output / "family_dry_run_v02.json",
        "coverage": output / "chain_token_coverage_dry_run.json",
    }
    paths.update({f"special:{p.name}": p for p in Path(args.special_dir).glob("*.tsv")})
    dump_json(
        output / "input_hashes.json",
        {
            name: {"path": str(path.resolve()), "sha256": file_hash(path)}
            for name, path in paths.items()
        },
    )
    dump_json(output / "preflight.json", preflight)
    print(
        json.dumps(
            {
                "family": summary["estimated_full_by_chemistry_family"],
                "planned": summary["estimated_full_structures"],
                "chain_coverage": chain_report["families"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
