import hashlib
import json
import platform
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import pyarrow as pa
import yaml
from rdkit import rdBase

from .assembly import assemble
from .budget import MandatoryBudgetExceeded, allocate_budget
from .chain_combinations import active_anchors, build_combination_bank
from .chain_library import import_chains, import_p_to_o
from .deduplicate import StructureStore
from .parquet_io import (
    CHAIN_SCHEMA,
    COMBINATION_SCHEMA,
    POLAR_SCHEMA,
    REJECTION_SCHEMA,
    STRUCTURE_SCHEMA,
    write_rows,
)
from .polar_library import import_polars
from .report import ReportAccumulator
from .sampling import StratifiedSampler
from .schemas import BuildConfig, LibraryValidationError, Rejection


def file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_library(
    config: BuildConfig,
    chains_path: str | Path,
    polar_path: str | Path,
    output: str | Path,
    mapping_path: str | Path | None = None,
):
    config.validate()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory is nonempty; choose a new directory to preserve existing builds"
        )
    output.mkdir(parents=True, exist_ok=True)
    paths = {"chains": Path(chains_path).resolve(), "polar": Path(polar_path).resolve()}
    if mapping_path is not None:
        paths["p_to_o_mapping"] = Path(mapping_path).resolve()
    resolved = {
        **asdict(config),
        "inputs": {
            name: {"path": str(path), "sha256": file_hash(path)} for name, path in paths.items()
        },
        "output": str(output),
    }
    (output / "build_config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    rejections, store = [], None
    try:
        chains = import_chains(paths["chains"], rejections)
        polars = import_polars(paths["polar"], rejections)
        if not chains or not polars:
            raise ValueError("Validated chain and polar libraries must both be nonempty")
        mapping = import_p_to_o(paths.get("p_to_o_mapping"), chains, rejections)
        anchors = active_anchors(chains, config, rejections)
        plan = allocate_budget(polars, anchors, config)
        bank = build_combination_bank(chains, config, mapping)
        write_rows(
            output / "chain_library.parquet", (c.row() for c in chains.values()), CHAIN_SCHEMA
        )
        write_rows(output / "polar_library.parquet", (p.row() for p in polars), POLAR_SCHEMA)
        write_rows(
            output / "chain_combinations.parquet", (c.row() for c in bank), COMBINATION_SCHEMA
        )
        for polar in polars:
            if not polar.allow_aa and anchors:
                rejections.append(
                    Rejection(
                        "anchors",
                        polar.polar_id,
                        "anchor_not_permitted",
                        "AA anchors are incompatible with external allow_aa=false",
                    )
                )
            if polar.polar_family == "PE" and polar.allow_pa and config.enable_pe_p:
                continue
            for token, chain in chains.items():
                if chain.linkage_type == "P" and token not in mapping:
                    rejections.append(
                        Rejection(
                            "p_policy",
                            f"{polar.polar_id}:{token}",
                            "p_to_o_mapping_missing",
                            "No independent P outside enabled PE-P; unmapped P skipped",
                        )
                    )
        store = StructureStore(output / "dedup.sqlite", config.max_structures)

        def add(polar, combo, mandatory=False):
            try:
                row = assemble(chains, combo, polar)
                row["is_anchor_combination"] = mandatory
                return store.add(row)
            except LibraryValidationError as exc:
                rejections.append(
                    Rejection(
                        "assembly", f"{polar.polar_id}:{combo.combination_id}", exc.reason, str(exc)
                    )
                )
                if mandatory:
                    raise ValueError(
                        f"Mandatory anchor failed: {polar.polar_id}:{combo.combination_id}"
                    ) from exc
                return False

        for polar in polars:
            if polar.allow_aa:
                for combo in anchors:
                    add(polar, combo, mandatory=True)
        mandatory_unique = store.count
        di_sampler = StratifiedSampler(bank, config, (combo.combination_id for combo in anchors))
        lyso_sampler = StratifiedSampler(bank, config)
        iterators, extra_counts, exhausted = {}, {p.polar_id: 0 for p in polars}, set()

        def extras(polar):
            lyso = lyso_sampler.lyso_candidates(polar)
            yield from islice(lyso, plan.lyso_quotas[polar.polar_id])
            yield from di_sampler.candidates(polar, lyso=False)
            yield from lyso

        for polar in polars:
            iterators[polar.polar_id] = iter(extras(polar))

        def fill(polar, quota):
            start = extra_counts[polar.polar_id]
            iterator = iterators[polar.polar_id]
            while (
                extra_counts[polar.polar_id] - start < quota and store.count < config.max_structures
            ):
                combo = next(iterator, None)
                if combo is None:
                    exhausted.add(polar.polar_id)
                    break
                if add(polar, combo):
                    extra_counts[polar.polar_id] += 1

        for polar in polars:
            fill(polar, plan.extra_quotas[polar.polar_id])
        while store.count < config.max_structures:
            eligible = [
                p
                for p in polars
                if p.polar_id not in exhausted
                and extra_counts[p.polar_id] < config.max_extra_per_head
            ]
            if not eligible:
                break
            before = store.count
            for polar in eligible:
                fill(polar, 1)
            if store.count == before:
                break
        accumulator = ReportAccumulator(polars)

        def export():
            for row in store.records():
                accumulator.add(row)
                yield row

        write_rows(
            output / "structures.parquet", export(), STRUCTURE_SCHEMA, config.parquet_batch_size
        )
        report = accumulator.finish(
            config, len(chains), len(bank), rejections, store.duplicates_removed
        )
        report.update(
            {
                "runtime_versions": {
                    "python": platform.python_version(),
                    "rdkit": rdBase.rdkitVersion,
                    "pyarrow": pa.__version__,
                },
                "sampling_method": "SHA256-ranked strata with head-specific rotations",
                "mandatory_anchor_count": plan.mandatory_anchor_count,
                "mandatory_unique_structures": mandatory_unique,
                "extra_quotas_initial": plan.extra_quotas,
                "extra_counts_actual": extra_counts,
                "p_to_o_mapping_provided": mapping_path is not None,
                "validated_p_to_o_mapping": mapping,
                "data_kind": "demo_fixture_not_real_training_library"
                if all("demo" in c.source for c in chains.values())
                else "external_supplied_fragment_tables",
                "polar_duplicate_sources_merged": sum(len(p.source_polar_ids) - 1 for p in polars),
            }
        )
        write_rows(output / "rejections.parquet", (asdict(r) for r in rejections), REJECTION_SCHEMA)
        (output / "library_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report
    except Exception as exc:
        failure = {
            "status": "failed",
            "training_performed": False,
            "error": str(exc),
            "total_structures": 0,
            "rejected_count": len(rejections),
        }
        if isinstance(exc, MandatoryBudgetExceeded):
            failure.update(
                {
                    "mandatory_anchor_count": exc.mandatory_anchor_count,
                    "max_allowed_eligible_polar_heads": exc.max_allowed_eligible_heads,
                }
            )
        write_rows(output / "rejections.parquet", (asdict(r) for r in rejections), REJECTION_SCHEMA)
        (output / "library_report.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    finally:
        if store is not None:
            store.close()
