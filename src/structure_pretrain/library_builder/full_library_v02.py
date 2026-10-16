"""v0.2 entry point kept separate from the frozen v0.1 builder and download pipeline."""

import csv
import json
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import yaml

from structure_pretrain.polar_plausibility.pipeline import dump_json
from structure_pretrain.polar_plausibility.reactive_rules import load_rules
from structure_pretrain.polar_plausibility.tiering import evaluate

from .assembly import assemble
from .canonicalize import nonisomeric_smiles, parse_molecule
from .chain_combinations import active_anchors, build_combination_bank, make_combination, permitted
from .chain_library import boolean, import_chains
from .deduplicate import StructureStore
from .family_sampling import allocate_family_budget, planned_extras
from .parquet_io import (
    CHAIN_SCHEMA,
    COMBINATION_SCHEMA,
    POLAR_SCHEMA,
    REJECTION_SCHEMA,
    STRUCTURE_SCHEMA,
    write_rows,
)
from .pipeline import file_hash
from .polar_library import import_polars
from .report import ReportAccumulator
from .sampling import StratifiedSampler, stable_number
from .schemas import BuildConfig, ChainSamplingConfig, LibraryValidationError, Rejection
from .special_cases_v02 import special_rule

PROVENANCE_FIELDS = [
    ("polar_plausibility_tier", pa.string()),
    ("chemistry_family", pa.string()),
    ("is_special_case", pa.bool_()),
    ("special_case_family", pa.string()),
    ("evidence_source", pa.string()),
    ("source_id", pa.string()),
    ("is_full_lipid_reported", pa.bool_()),
    ("is_programmatically_assembled", pa.bool_()),
    ("notes", pa.string()),
    ("structure_origin", pa.string()),
    ("source_cids", pa.list_(pa.int64())),
    ("parent_source_cids", pa.list_(pa.int64())),
    ("source_attachment_provenance", pa.string()),
    ("also_generated_as_phosphate_analog", pa.bool_()),
    ("analog_parent_polar_ids", pa.list_(pa.string())),
]
STRUCTURES_V02 = pa.schema(list(STRUCTURE_SCHEMA) + [pa.field(*f) for f in PROVENANCE_FIELDS])
POLARS_V02 = pa.schema(
    list(POLAR_SCHEMA) + [pa.field(*f) for f in PROVENANCE_FIELDS if f[0] != "source_id"]
)


@dataclass(frozen=True)
class AnchorOptions:
    adaptive_anchor_threshold: float = 0.90
    mandatory_common_anchor_count: int = 3
    optional_anchor_budget_fraction: float = 0.35
    family_sampling_weight: dict[str, float] | None = None


def load_config_v02(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    options = AnchorOptions(
        **{name: raw.pop(name) for name in AnchorOptions.__dataclass_fields__ if name in raw}
    )
    sampling = ChainSamplingConfig(**raw.pop("chain_sampling", {}))
    config = BuildConfig(**raw, chain_sampling=sampling)
    config.validate()
    if config.schema_version != "structure-library-v0.2":
        raise ValueError("v0.2 configuration required")
    if not (
        0 < options.adaptive_anchor_threshold <= 1
        and 0 <= options.optional_anchor_budget_fraction <= 1
        and options.mandatory_common_anchor_count >= 3
    ):
        raise ValueError("Invalid adaptive anchor options")
    if options.family_sampling_weight is not None:
        from math import isfinite

        if set(options.family_sampling_weight) != {
            "phosphate",
            "phosphonate",
            "polyphosphate",
        } or any(not isfinite(v) or v <= 0 for v in options.family_sampling_weight.values()):
            raise ValueError("All three chemistry families need finite positive sampling weights")
    return config, options


def read_tsv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def list_field(row, name):
    value = row.get(name) or []
    return json.loads(value) if isinstance(value, str) else value


def metadata(row):
    family = row.get("chemistry_family", row["polar_family"])
    if row.get("special_case_family") in {"keto_inositol", "PI_reference"}:
        family = "phosphate"
    return {
        "polar_plausibility_tier": row["polar_plausibility_tier"],
        "chemistry_family": family,
        "is_special_case": boolean(row.get("is_special_case"), False),
        "special_case_family": row.get("special_case_family", ""),
        "evidence_source": row.get("evidence_source") or row["source"],
        "source_id": row["source_id"],
        # Head-level source records alone do not verify a full lipid graph.
        "is_full_lipid_reported": False,
        "is_programmatically_assembled": True,
        "notes": row.get("notes", ""),
        "structure_origin": row.get("structure_origin")
        or (
            "generated_keto_inositol"
            if row["source"] == "assembled_from_known_polar"
            else "pubchem_baseline"
            if row["source"] == "PubChem"
            else "external_supplied"
        ),
        "source_cids": list_field(row, "source_cids"),
        "parent_source_cids": list_field(row, "parent_source_cids"),
        "source_attachment_provenance": json.dumps(
            list_field(row, "source_attachment_provenance"), sort_keys=True
        ),
        "also_generated_as_phosphate_analog": boolean(
            row.get("also_generated_as_phosphate_analog"), False
        ),
        "analog_parent_polar_ids": list_field(row, "analog_parent_polar_ids"),
    }


def load_approved_v02(
    approved_path,
    report_path,
    special_dir=None,
    plausibility_config="configs/polar_plausibility.yaml",
    reactive_config="configs/polar_reactive_rules.yaml",
):
    approved_path, report_path = Path(approved_path), Path(report_path)
    if not approved_path.is_file() or not report_path.is_file():
        raise ValueError("Formal assembly blocked: CORE + EXTENDED table/report not generated")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or file_hash(approved_path) != report.get(
        "approved_table_sha256"
    ):
        raise ValueError("Approved input must match a completed plausibility report hash")
    records = read_tsv(approved_path)
    if len(records) != report["approved_polar_count"]:
        raise ValueError("Approved polar count disagrees with filtering report")
    if any(r.get("polar_plausibility_tier") not in {"CORE", "EXTENDED"} for r in records):
        raise ValueError("Only CORE + EXTENDED can enter formal assembly")
    errors = []
    polars = import_polars(approved_path, errors)
    if errors or len(polars) != len(records):
        raise ValueError("Approved table contains invalid/conflicting/duplicate polar rows")
    details = {r["polar_id"]: metadata(r) for r in records}
    by_key = {nonisomeric_smiles(parse_molecule(p.polar_fragment_smiles)): p for p in polars}
    required_combos = {}
    if special_dir is not None:
        special_dir = Path(special_dir)
        must = {r["polar_id"] for r in read_tsv(special_dir / "must_include_polar_ids.tsv")}
        supplemental = read_tsv(special_dir / "special_case_polars.tsv")
        config = yaml.safe_load(Path(plausibility_config).read_text(encoding="utf-8"))
        rules = load_rules(reactive_config)
        aliases = {}
        for row in supplemental:
            key = nonisomeric_smiles(parse_molecule(row["polar_fragment_smiles"]))
            evaluation = evaluate(
                {"status": "APPROVABLE", "polar_fragment_smiles": key}, config, rules
            )
            if evaluation["tier"] not in {"CORE", "EXTENDED"}:
                raise ValueError(
                    f"Special case failed identical chemistry rules: {row['polar_id']}"
                )
            current = by_key.get(key)
            if current is None:
                # Only graph-generated keto candidates may supplement the real filtered table.
                if (
                    row["evidence_source"] != "assembled_from_known_polar"
                    or row["special_case_family"] != "keto_inositol"
                ):
                    raise ValueError("Real PubChem special polar must already be approved")
                current = special_rule(
                    key, row["polar_id"], row["source_id"], row["evidence_source"]
                )
                polars.append(current)
                by_key[key] = current
                row["polar_plausibility_tier"] = evaluation["tier"]
                details[current.polar_id] = metadata(row)
            aliases[row["polar_id"]] = current.polar_id
            if row["polar_id"] in must:
                details[current.polar_id].update(
                    is_special_case=True,
                    special_case_family=row["special_case_family"],
                    notes=row["notes"],
                )
        if not must.issubset(aliases):
            raise ValueError("Missing must-include special head")
        for name in ("nitrile_special_cases.tsv", "keto_pi_special_cases.tsv"):
            for case in read_tsv(special_dir / name):
                identifier = aliases[case["polar_id"]]
                required_combos.setdefault(identifier, set()).add(case["chains"])
    return sorted(polars, key=lambda p: p.polar_id), details, required_combos


def plan_anchors(polars, anchors, config, options, required_combos=None):
    if len(anchors) < options.mandatory_common_anchor_count:
        raise ValueError("At least three configured common AA anchors must be available")
    adaptive = (
        len(polars) * len(anchors) >= config.max_structures * options.adaptive_anchor_threshold
    )
    family_aware = options.family_sampling_weight is not None
    common = (
        anchors[: options.mandatory_common_anchor_count] if adaptive or family_aware else anchors
    )
    selected = {p.polar_id: {c.combination_id for c in common} for p in polars}
    if any(not p.allow_aa for p in polars):
        raise ValueError("v0.2 requires shared AA backgrounds for every polar")
    for identifier, combos in (required_combos or {}).items():
        if identifier not in selected:
            raise ValueError("Required special-case polar absent")
        selected[identifier].update(combos)
    mandatory_count = sum(len(combos) for combos in selected.values())
    if mandatory_count > config.max_structures:
        raise ValueError(
            "Even three shared anchors plus required specials exceed budget; no heads deleted"
        )
    optional_remaining = int(
        (config.max_structures - mandatory_count) * options.optional_anchor_budget_fraction
    )
    if adaptive and not family_aware:
        optional = anchors[options.mandatory_common_anchor_count :]
        # Cycle heads in a seed-derived order, rotating anchor choice independently per head.
        order = sorted(polars, key=lambda p: stable_number(f"{config.seed}|optional|{p.polar_id}"))
        for turn in range(len(optional)):
            for polar in order:
                if not optional_remaining:
                    break
                offset = stable_number(f"{config.seed}|{polar.polar_id}") % max(1, len(optional))
                label = optional[(offset + turn) % len(optional)].combination_id
                if label not in selected[polar.polar_id]:
                    selected[polar.polar_id].add(label)
                    optional_remaining -= 1
    return selected, {
        "adaptive_anchors": adaptive,
        "anchor_policy": "three_common_then_family_weighted_extras"
        if family_aware
        else "adaptive_shared_anchors",
        "common_anchor_ids": [c.combination_id for c in common],
        "required_before_optional_rotation": mandatory_count,
        "selected_anchor_and_special_structures": sum(len(v) for v in selected.values()),
    }


def preflight_v02(config, options, chains_path, approved_path, report_path, special_dir=None):
    errors = []
    chains = import_chains(chains_path, errors)
    if errors or Counter(c.linkage_type for c in chains.values()) != {"A": 177, "O": 84, "P": 61}:
        raise ValueError("Formal build requires the frozen A177/O84/P61 chain table")
    polars, details, required = load_approved_v02(approved_path, report_path, special_dir)
    anchors = active_anchors(chains, config, errors)
    if errors:
        raise ValueError(
            "An anchor chain is missing; frozen input does not meet v0.2 specification"
        )
    selected, plan = plan_anchors(polars, anchors, config, options, required)
    bank = build_combination_bank(chains, config)
    report = {
        "status": "preflight_complete_no_assembly",
        "polar_count": len(polars),
        "real_filtered_polar_count": json.loads(Path(report_path).read_text(encoding="utf-8")).get(
            "empirical_polar_count",
            json.loads(Path(report_path).read_text(encoding="utf-8"))["approved_polar_count"],
        ),
        "generated_supplemental_head_count": sum(
            p.source == "assembled_from_known_polar" for p in polars
        ),
        "polar_tier_counts": dict(Counter(d["polar_plausibility_tier"] for d in details.values())),
        "chain_count": len(chains),
        "chain_combination_count": len(bank),
        **plan,
        "remaining_budget": config.max_structures - sum(map(len, selected.values())),
        "max_structures": config.max_structures,
        "training_performed": False,
        "full_lipid_library_generated": False,
    }
    return report, (chains, polars, details, selected, bank)


def build_v02(
    config,
    options,
    chains_path,
    approved_path,
    report_path,
    output,
    special_dir=None,
    scratch_dir=None,
):
    report, (chains, polars, details, selected, bank) = preflight_v02(
        config, options, chains_path, approved_path, report_path, special_dir
    )
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve existing builds: output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "chains": Path(chains_path),
        "approved_polar": Path(approved_path),
        "plausibility_report": Path(report_path),
    }
    if special_dir:
        for path in sorted(Path(special_dir).glob("*.tsv")):
            input_paths[f"special:{path.name}"] = path
    resolved = {
        **asdict(config),
        **asdict(options),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": file_hash(path)}
            for name, path in input_paths.items()
        },
    }
    (output / "build_config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    write_rows(output / "chain_library.parquet", (c.row() for c in chains.values()), CHAIN_SCHEMA)
    write_rows(
        output / "polar_library.parquet",
        ({**p.row(), **details[p.polar_id]} for p in polars),
        POLARS_V02,
    )
    write_rows(output / "chain_combinations.parquet", (c.row() for c in bank), COMBINATION_SCHEMA)
    by_label = {combo.combination_id: combo for combo in bank}
    rejections = []
    with tempfile.TemporaryDirectory(prefix="lipid-v02-", dir=scratch_dir) as scratch:
        store = StructureStore(Path(scratch) / "dedup.sqlite", config.max_structures)
        try:

            def add(polar, combo, mandatory=False):
                try:
                    if not permitted(combo, polar, config):
                        raise LibraryValidationError(
                            "chain_policy", "Required combination not permitted"
                        )
                    row = assemble(chains, combo, polar)
                    row.update(details[polar.polar_id])
                    row["is_anchor_combination"] = (
                        combo.combination_id in config.anchor_combinations
                    )
                    return store.add(row)
                except LibraryValidationError as exc:
                    rejections.append(Rejection("assembly", polar.polar_id, exc.reason, str(exc)))
                    if mandatory:
                        raise ValueError(
                            f"Required structure failed: {polar.polar_id}/{combo.combination_id}"
                        ) from exc
                    return False

            for polar in polars:
                for label in sorted(selected[polar.polar_id]):
                    if label not in by_label:
                        by_label[label] = make_combination(tuple(label.split("_")), chains)
                    add(polar, by_label[label], mandatory=True)
            initial_unique = store.count
            sampler = StratifiedSampler(bank, config)
            remaining = config.max_structures - store.count
            base, remainder = divmod(remaining, len(polars))
            total_extra_target = sum(
                min(config.max_extra_per_head, base + (i < remainder)) for i in range(len(polars))
            )
            lyso_remaining = min(
                total_extra_target, round(config.max_structures * config.lyso_fraction_target)
            )
            extra_counts = Counter()
            if options.family_sampling_weight is not None:
                budget = allocate_family_budget(
                    polars, details, selected, config, options.family_sampling_weight
                )
                for polar in polars:
                    for combo in planned_extras(polar, selected[polar.polar_id], budget, sampler):
                        if add(polar, combo):
                            extra_counts[polar.polar_id] += 1
            else:
                # Preserve the earlier uniform v0.2 schedule for its existing configuration.
                for index, polar in enumerate(polars):
                    quota = min(config.max_extra_per_head, base + (index < remainder))
                    lyso_quota = min(
                        quota, (lyso_remaining + len(polars) - index - 1) // (len(polars) - index)
                    )
                    used = 0
                    for combo in sampler.lyso_candidates(polar):
                        if used >= lyso_quota or store.count >= config.max_structures:
                            break
                        if combo.combination_id not in selected[polar.polar_id] and add(
                            polar, combo
                        ):
                            used += 1
                    lyso_remaining -= used
                    extra_counts[polar.polar_id] += used
                    for combo in sampler.candidates(polar, lyso=False):
                        if (
                            extra_counts[polar.polar_id] >= quota
                            or store.count >= config.max_structures
                        ):
                            break
                        if combo.combination_id not in selected[polar.polar_id] and add(
                            polar, combo
                        ):
                            extra_counts[polar.polar_id] += 1
            accumulator = ReportAccumulator(polars)
            special_rows, special_counts = [], Counter()
            per_head_anchors = Counter()

            def export():
                for row in store.records():
                    accumulator.add(row)
                    per_head_anchors[row["polar_id"]] += row["is_anchor_combination"]
                    if row["is_special_case"]:
                        special_rows.append(row)
                        special_counts[row["special_case_family"]] += 1
                    yield row

            write_rows(
                output / "structures.parquet", export(), STRUCTURES_V02, config.parquet_batch_size
            )
            write_rows(output / "special_cases.parquet", special_rows, STRUCTURES_V02)
            write_rows(
                output / "rejections.parquet", (asdict(r) for r in rejections), REJECTION_SCHEMA
            )
            actual = accumulator.finish(
                config, len(chains), len(bank), rejections, store.duplicates_removed
            )
            if actual["structures_per_polar_min"] < 3:
                raise ValueError("Coverage failed: some heads lost their three common backgrounds")
            actual.update(
                {
                    "polar_tier_counts": report["polar_tier_counts"],
                    "real_filtered_polar_count": report["real_filtered_polar_count"],
                    "generated_supplemental_head_count": report[
                        "generated_supplemental_head_count"
                    ],
                    "structures_per_polar_mean": store.count / len(polars),
                    "initial_required_unique_structures": initial_unique,
                    "anchor_plan": {
                        k: report[k]
                        for k in (
                            "adaptive_anchors",
                            "common_anchor_ids",
                            "required_before_optional_rotation",
                            "selected_anchor_and_special_structures",
                        )
                    },
                    "anchor_coverage_min": min(per_head_anchors.values(), default=0),
                    "special_full_structure_counts": dict(special_counts),
                    "database_literature_exact_full_lipid_hits": sum(
                        r["is_full_lipid_reported"] for r in special_rows
                    ),
                    "programmatically_assembled_special_structures": len(special_rows),
                    "within_1500000": store.count <= 1_500_000,
                    "within_2000000": store.count <= 2_000_000,
                    "within_configured_budget": store.count <= config.max_structures,
                    "chemistry_family_used_for_extra_budget": options.family_sampling_weight
                    is not None,
                    "family_sampling_weight": options.family_sampling_weight,
                    "full_lipid_library_generated": True,
                    "provenance_used_for_loss_sampling_or_labels": False,
                }
            )
            dump_json(output / "library_report.json", actual)
            return actual
        except Exception as exc:
            dump_json(
                output / "library_report.json",
                {
                    "status": "failed",
                    "error": str(exc),
                    "training_performed": False,
                    "full_lipid_library_generated": False,
                },
            )
            raise
        finally:
            store.close()
