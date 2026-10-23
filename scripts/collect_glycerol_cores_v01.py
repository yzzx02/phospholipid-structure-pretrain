"""CPU-only future-core library, distinct from active V2 frozen training."""

import argparse
import json
from pathlib import Path

from structure_pretrain.glycerol_core_collection.collection import annotate_frozen, collect
from structure_pretrain.glycerol_core_collection.finalize import finalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/glycerol_core_collection_v01.json")
    p.add_argument("--skip-frozen-sidecar", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not (
        config["mass_limit_glycosyl_or_inositol_Da"] == 600
        and config["mass_limit_other_Da"] == 400
        and config["extra_head_chain_min_contiguous_aliphatic_carbons"] == 6
        and config["inositol_in_600_group"]
        and not config["retain_stereoisomers_as_separate_templates"]
    ):
        raise ValueError(
            "v0.1 chemical policy is frozen; update and validate the checker for a new scope"
        )
    if args.finalize_only:
        print(json.dumps(finalize(config["root"])["selection_status_counts"], ensure_ascii=False))
        return
    groups, summary = collect(config)
    if not args.skip_frozen_sidecar:
        summary["frozen_library_sidecar"] = annotate_frozen(config, groups)
        Path(config["root"], "collection_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = finalize(config["root"])
    print(
        json.dumps(
            {"complete": True, "unique_core_graphs": summary["unique_connectivity_core_graphs"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
