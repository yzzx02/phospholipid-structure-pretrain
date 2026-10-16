from collections import Counter
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import (
    active_anchors,
    build_combination_bank,
)
from structure_pretrain.library_builder.chain_library import import_chains
from structure_pretrain.library_builder.coverage import chain_coverage
from structure_pretrain.library_builder.family_sampling import dry_run_family_schedule
from structure_pretrain.library_builder.full_library_v02 import (
    AnchorOptions,
    metadata,
    plan_anchors,
)
from structure_pretrain.library_builder.polar_library import import_polars
from structure_pretrain.library_builder.production import (
    assemble_shard,
    cached_fragment,
    combine_shards,
    init_assembly,
    scheduled_jobs,
    validate_formula,
)
from structure_pretrain.library_builder.schemas import LibraryValidationError, load_build_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/library_builder"
WEIGHTS = {"phosphate": 2.0, "phosphonate": 1.0, "polyphosphate": 1.0}


def test_user_policy_p_coverage_is_descriptive_but_ao_is_required():
    chains = import_chains(EXAMPLES / "chains.tsv", [])
    counts = {t: 1 for t, c in chains.items() if c.linkage_type != "P"}
    report = chain_coverage(chains, counts)
    assert report["status"] == "passed"
    assert report["families"]["P"]["covered"] == 0
    assert not report["families"]["P"]["coverage_is_hard_requirement"]
    counts["A-16:0"] = 0
    assert chain_coverage(chains, counts)["status"] == "failed"


def test_real_shards_resume_global_dedup_and_formula_audit(tmp_path):
    import hashlib
    from dataclasses import asdict

    chains = import_chains(EXAMPLES / "chains.tsv", [])
    polars = sorted(import_polars(EXAMPLES / "polar_moieties.tsv", []), key=lambda p: p.polar_id)
    config = replace(
        load_build_config(EXAMPLES / "demo_build.yaml"),
        schema_version="structure-library-v0.2",
        max_structures=100,
        max_extra_per_head=100,
    )
    options = AnchorOptions(family_sampling_weight=WEIGHTS)
    details = {
        p.polar_id: metadata(
            {
                **p.row(),
                "polar_plausibility_tier": "CORE",
                "chemistry_family": "phosphonate"
                if p.polar_family == "phosphonate"
                else "phosphate",
            }
        )
        for p in polars
    }
    selected, _ = plan_anchors(polars, active_anchors(chains, config, []), config, options)
    bank = build_combination_bank(chains, config)
    dry, _, _ = dry_run_family_schedule(polars, details, selected, bank, config, WEIGHTS)
    dry["special_head_background_counts"] = {}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**asdict(config), **asdict(options)}))
    init_assembly(EXAMPLES / "chains.tsv", path)
    digest = hashlib.sha256()
    jobs = list(
        scheduled_jobs(polars, details, selected, bank, config, WEIGHTS, tmp_path, digest, 2)
    )
    assert digest.hexdigest() == dry["schedule_sha256"]
    stats = [assemble_shard(job) for job in jobs]
    assert assemble_shard(jobs[0]) == stats[0]
    assert sum(s["planned"] for s in stats) == 100
    output = tmp_path / "output"
    output.mkdir()
    report = combine_shards(output, tmp_path, stats, chains, polars, selected, config, dry)
    rows = pq.read_table(output / "structures.parquet").to_pylist()
    assert report["status"] == "complete" and len(rows) == 100
    assert len({r["canonical_smiles"] for r in rows}) == len(rows)
    assert report["covered_polar_count"] == len(polars)
    assert all(len(r["morgan_fp_2048"]) == 256 for r in rows)
    counts = Counter(t for r in rows for t in set(r["chain_tokens"]))
    assert chain_coverage(chains, counts)["status"] == "passed"
    for row in rows:
        validate_formula(row)
        if any(t.startswith("P-") for t in row["chain_tokens"]):
            assert row["polar_family"] == "PE"
    broken = dict(rows[0], neutral_formula="C999")
    with pytest.raises(LibraryValidationError, match="disagree"):
        validate_formula(broken)
    # A cached fragment must remain unchanged across repeated graph assembly.
    combo = bank[0]
    first = assemble(chains, combo, polars[0])
    assert assemble(chains, combo, polars[0], fragment_parser=cached_fragment) == first
    assert assemble(chains, combo, polars[0], fragment_parser=cached_fragment) == first
