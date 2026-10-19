"""Append-only v0.3 extension of an audited library; never rebuild the baseline."""

import hashlib
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.polar_plausibility.pipeline import dump_json

from .assembly import assemble
from .chain_combinations import make_combination, permitted
from .chain_library import import_chains
from .coverage import chain_coverage
from .full_library_v02 import PROVENANCE_FIELDS, load_config_v02
from .parquet_io import REJECTION_SCHEMA, write_rows
from .pipeline import file_hash
from .production import FINAL_SCHEMA, cached_fragment, validate_formula
from .sampling import stable_number
from .schemas import LibraryValidationError, PolarRule


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze_files(directories):
    """Hash files, not directory mtimes. Call again after all derived work."""
    return {
        str(p.resolve()): file_hash(p)
        for directory in directories
        for p in sorted(Path(directory).rglob("*"))
        if p.is_file()
    }


def verify_frozen(manifest):
    changed = [p for p, digest in manifest.items() if file_hash(Path(p)) != digest]
    if changed:
        raise ValueError(f"Frozen baseline files changed: {changed}")


def resolve_input(entry, root):
    path = Path(entry["path"])
    if not path.is_file():
        # Some historical JSON paths were decoded through a Windows code page.
        # Resolve the ASCII repository suffix; preserve the historical JSON bytes.
        parts = entry["path"].replace("\\", "/").split("/")
        if root.name not in parts:
            raise ValueError(f"Cannot resolve frozen input: {entry['path']}")
        path = root.joinpath(*parts[parts.index(root.name) + 1 :])
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Frozen input hash differs: {path}")
    return {"path": str(path.resolve()), "sha256": entry["sha256"]}


def chain_stratum(row, config):
    carbon = row.get("total_carbon", row.get("total_chain_carbon"))
    db = row.get("total_double_bonds", row.get("total_chain_double_bonds"))
    bins = config.chain_sampling
    length = (
        0 if carbon <= bins.short_max_total_c else 1 if carbon <= bins.medium_max_total_c else 2
    )
    unsat = 0 if db <= bins.low_max_total_db else 1 if db <= bins.medium_max_total_db else 2
    family = row["linkage_family"]
    return family, length, unsat if family == "AA" else 0


def low_percentile(values):
    """Midrank empirical percentile, reversed so smaller values have higher priority."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    before = np.cumsum(counts) - counts
    return 1 - (before[inverse] + counts[inverse] / 2) / len(values)


def select_pairs(library, metrics_path, config, planned=15000):
    """One new background per selected head; rank deficits without family quotas."""
    library = Path(library)
    heads = pq.read_table(library / "polar_library.parquet", columns=["polar_id"])[
        "polar_id"
    ].to_pylist()
    head_code = {p: i for i, p in enumerate(heads)}
    bank = {
        r["combination_id"]: r
        for r in pq.read_table(library / "chain_combinations.parquet").to_pylist()
    }
    combo_codes = {label: i for i, label in enumerate(bank)}
    strata = {key: chain_stratum(r, config) for key, r in bank.items()}
    stratum_codes = {key: i for i, key in enumerate(sorted(set(strata.values())))}
    combo_strata = {key: stratum_codes[value] for key, value in strata.items()}
    count = np.zeros(len(heads), dtype=np.uint16)
    masks = np.zeros(len(heads), dtype=np.uint64)
    similarity_sum = np.zeros(len(heads), dtype=np.float64)
    metrics = np.load(metrics_path, allow_pickle=False)
    maximum = metrics["max_similarity"]
    source = pq.ParquetFile(library / "structures.parquet")
    if len(maximum) != source.metadata.num_rows:
        raise ValueError("Retrieval metrics and baseline row order differ")
    # At most 15 backgrounds/head in the current library: a fixed numeric matrix
    # avoids keeping two million Python strings and per-head Python sets in RAM.
    max_backgrounds = read_json(library / "library_report.json")["structures_per_polar_max"]
    used = np.full((len(heads), max_backgrounds), -1, dtype=np.int32)
    offset = 0
    for batch in source.iter_batches(batch_size=10000, columns=["polar_id", "combination_id"]):
        rows = batch.to_pydict()
        for j, (head, label) in enumerate(zip(rows["polar_id"], rows["combination_id"])):
            i = head_code[head]
            used[i, count[i]] = combo_codes[label]
            count[i] += 1
            masks[i] |= np.uint64(1) << np.uint64(combo_strata[label])
            similarity_sum[i] += maximum[offset + j]
        offset += batch.num_rows
    diversity = np.array([int(n).bit_count() for n in masks], dtype=np.uint8)
    mean_max = similarity_sum / count
    priority = (
        0.4 * low_percentile(count)
        + 0.3 * low_percentile(diversity)
        + 0.3 * low_percentile(mean_max)
    )
    order = sorted(
        range(len(heads)),
        key=lambda i: (-priority[i], stable_number(f"{config.seed}|increment|{heads[i]}")),
    )[: min(planned, len(heads))]
    selected = set(heads[i] for i in order)
    allowed = {}
    for batch in pq.ParquetFile(library / "polar_library.parquet").iter_batches(
        batch_size=1000, columns=["polar_id", "allow_aa", "allow_oa"]
    ):
        for row in batch.to_pylist():
            if row["polar_id"] in selected:
                allowed[row["polar_id"]] = row
    by_stratum = defaultdict(list)
    # No P additions: exact PE-P/A baseline rows remain untouched.
    for label, row in bank.items():
        if row["linkage_family"] not in {"PA", "P"}:
            by_stratum[combo_strata[label]].append(label)
    for labels in by_stratum.values():
        labels.sort(key=lambda label: stable_number(f"{config.seed}|increment|{label}"))
    labels_by_code = list(bank)
    selection = []
    for i in order:
        head = heads[i]
        old_labels = [labels_by_code[c] for c in used[i] if c >= 0]
        old = set(old_labels)
        old_tokens = {t for label in old for t in bank[label]["chain_tokens"]}
        old_strata = Counter(combo_strata[label] for label in old)
        candidates = []
        for s, labels in by_stratum.items():
            family = bank[labels[0]]["linkage_family"]
            if not allowed[head]["allow_aa" if family in {"AA", "A"} else "allow_oa"]:
                continue
            start = stable_number(f"{config.seed}|{head}|increment|{s}") % len(labels)
            # Bounded deterministic candidates from each bin, no full head x bank product.
            for shift in range(min(12, len(labels))):
                label = labels[(start + shift) % len(labels)]
                if label in old:
                    continue
                novelty = len(set(bank[label]["chain_tokens"]) - old_tokens)
                key = (
                    old_strata[s] == 0,
                    -old_strata[s],
                    novelty,
                    stable_number(f"{config.seed}|{head}|{label}"),
                )
                candidates.append((key, label))
        if candidates:
            label = max(candidates)[1]
            selection.append(
                {
                    "polar_id": head,
                    "combination_id": label,
                    "baseline_background_count": int(count[i]),
                    "baseline_chain_strata": int(diversity[i]),
                    "baseline_mean_max_similarity": float(mean_max[i]),
                    "priority_score": float(priority[i]),
                    "new_chain_stratum": combo_strata[label] not in old_strata,
                    "novel_chain_tokens": len(set(bank[label]["chain_tokens"]) - old_tokens),
                    "selection_reason": (
                        "0.4 low coverage + 0.3 low chain strata + 0.3 low similarity"
                    ),
                }
            )
    stats = {
        "policy": "One unused pair per prioritized head; no family or P quotas; no failure refill",
        "requested_proposals": planned,
        "selected_unused_pairs": len(selection),
        "selected_heads": len(selection),
        "novel_chain_strata_pairs": sum(r["new_chain_stratum"] for r in selection),
        "new_chain_token_pairs": sum(r["novel_chain_tokens"] > 0 for r in selection),
        "all_heads": {
            "background_count_mean": float(count.mean()),
            "chain_strata_mean": float(diversity.mean()),
            "mean_max_similarity": float(mean_max.mean()),
        },
        "selected_heads_before_extension": {
            "background_count_mean": float(
                np.mean([r["baseline_background_count"] for r in selection])
            ),
            "chain_strata_mean": float(np.mean([r["baseline_chain_strata"] for r in selection])),
            "mean_max_similarity": float(
                np.mean([r["baseline_mean_max_similarity"] for r in selection])
            ),
        },
    }
    return selection, stats


def accept_unique(candidates, baseline_db, duplicate_rows):
    """Read-only baseline key lookup and full canonical-string increment dedup."""
    seen = set()
    with sqlite3.connect(Path(baseline_db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        for row in candidates:
            key = row["canonical_smiles"]
            duplicate = (
                key in seen
                or conn.execute("SELECT 1 FROM seen WHERE smiles=?", (key,)).fetchone() is not None
            )
            if duplicate:
                duplicate_rows.append(
                    {k: row[k] for k in ("structure_id", "polar_id", "combination_id")}
                )
            else:
                seen.add(key)
                yield row


def append_baseline(baseline, increment, destination):
    """Copy all original columns and values in their existing order, then append."""
    original = pq.ParquetFile(baseline)
    with pq.ParquetWriter(destination, original.schema_arrow, compression="zstd") as writer:
        for path in (baseline, increment):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4000):
                writer.write_batch(batch)


def verify_prefix(baseline, combined):
    new_batches = iter(pq.ParquetFile(combined).iter_batches(batch_size=4000))
    checked = 0
    for original in pq.ParquetFile(baseline).iter_batches(batch_size=4000):
        other = next(new_batches).slice(0, original.num_rows)
        if not original.equals(other):
            raise ValueError(f"Baseline prefix changed at row {checked}")
        checked += original.num_rows
    return checked


def build_increment(library, cache, output, baseline_db, chains_path, config_path, planned=15000):
    library, cache, output = Path(library), Path(cache), Path(output)
    root = Path.cwd()
    if output.exists() or output.resolve() == library.resolve():
        raise ValueError("Increment requires a new output directory; overwriting is prohibited")
    base_report = read_json(library / "library_report.json")
    audit = read_json(library / "verification_audit.json")
    manifest = read_json(cache / "fingerprint_manifest.json")
    if base_report["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Audited baseline required")
    digest = file_hash(library / "structures.parquet")
    if digest != audit["structures_sha256"] or digest != manifest["structures_sha256"]:
        raise ValueError("Baseline audit/cache do not bind to this exact library")
    frozen = freeze_files([library, cache])
    frozen[str(Path(baseline_db).resolve())] = file_hash(Path(baseline_db))
    resolved = {
        k: resolve_input(v, root) for k, v in read_json(library / "input_hashes.json").items()
    }
    chains = import_chains(chains_path, [])
    if file_hash(Path(chains_path)) != resolved["chains"]["sha256"]:
        raise ValueError("Only the frozen original chain representatives may be used")
    if (
        "final_config" in resolved
        and file_hash(Path(config_path)) != resolved["final_config"]["sha256"]
    ):
        raise ValueError("Increment chemistry policy must use the frozen final baseline config")
    config, _ = load_config_v02(config_path)
    output.mkdir(parents=True)
    dump_json(output / "baseline_preservation_manifest.json", frozen)
    dump_json(output / "resolved_frozen_inputs.json", resolved)
    selection, selection_stats = select_pairs(
        library, cache / "target_readiness_metrics.npz", config, planned
    )
    pq.write_table(
        pa.Table.from_pylist(selection), output / "increment_selection.parquet", compression="zstd"
    )
    dump_json(output / "selection_report.json", selection_stats)
    print(f"increment selected {len(selection)} unused pairs", flush=True)
    desired = {r["polar_id"]: r for r in selection}
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    rejections, duplicates = [], []
    counters = {key: Counter() for key in ("family", "origin", "tier", "linkage", "head", "token")}
    attempted = assembled = 0

    def candidates():
        nonlocal attempted, assembled
        for batch in pq.ParquetFile(library / "polar_library.parquet").iter_batches(batch_size=500):
            for record in batch.to_pylist():
                if record["polar_id"] not in desired:
                    continue
                polar = PolarRule(**{key: record[key] for key in PolarRule.__dataclass_fields__})
                label = desired[polar.polar_id]["combination_id"]
                attempted += 1
                try:
                    combo = make_combination(tuple(label.split("_")), chains)
                    if combo.linkage_family in {"PA", "P"} or not permitted(combo, polar, config):
                        raise LibraryValidationError("chain_policy", "Unpermitted incremental pair")
                    row = assemble(chains, combo, polar, fragment_parser=cached_fragment)
                    validate_formula(row)
                    row.update({key: record[key] for key, _ in PROVENANCE_FIELDS})
                    row["structure_id"] = (
                        "pl_" + hashlib.sha256(row["canonical_smiles"].encode()).hexdigest()
                    )
                    row["polar_fragment_smiles"] = polar.polar_fragment_smiles
                    row["is_anchor_combination"] = label in config.anchor_combinations
                    row["is_mandatory_pair"] = False
                    row["morgan_fp_2048"] = DataStructs.BitVectToBinaryText(
                        generator.GetFingerprint(cached_fragment(row["canonical_smiles"]))
                    )
                    assembled += 1
                    yield row
                except LibraryValidationError as exc:
                    rejections.append(
                        {
                            "stage": "increment",
                            "record_id": f"{polar.polar_id}/{label}",
                            "reason": exc.reason,
                            "message": str(exc),
                        }
                    )
                if attempted % 1000 == 0:
                    print(f"increment assembled {attempted}/{len(selection)}", flush=True)

    def rows():
        for row in accept_unique(candidates(), baseline_db, duplicates):
            for name, field in (
                ("family", "chemistry_family"),
                ("origin", "structure_origin"),
                ("tier", "polar_plausibility_tier"),
                ("head", "polar_id"),
            ):
                counters[name][row[field]] += 1
            counters["linkage"]["lyso" if row["is_lyso"] else row["linkage_family"]] += 1
            counters["token"].update(set(row["chain_tokens"]))
            yield row

    with rdBase.BlockLogs():
        write_rows(output / "increment_structures.parquet", rows(), FINAL_SCHEMA)
    unique = sum(counters["head"].values())
    append_baseline(
        library / "structures.parquet",
        output / "increment_structures.parquet",
        output / "structures.parquet",
    )
    preserved = verify_prefix(library / "structures.parquet", output / "structures.parquet")
    write_rows(output / "increment_rejections.parquet", rejections, REJECTION_SCHEMA)
    write_rows(
        output / "increment_canonical_duplicates.parquet",
        duplicates,
        pa.schema([(key, pa.string()) for key in ("structure_id", "polar_id", "combination_id")]),
    )
    for name in (
        "polar_library.parquet",
        "chain_library.parquet",
        "chain_combinations.parquet",
        "input_hashes.json",
        "build_config_resolved.yaml",
    ):
        shutil.copyfile(library / name, output / name)
    # Combined summary is derived arithmetically from the frozen audit + successful increment.
    report = dict(base_report)
    report.update(
        schema_version="structure-library-v0.3",
        baseline_unique_full_structures=preserved,
        increment_planned=len(selection),
        increment_attempted=attempted,
        increment_assembled=assembled,
        increment_unique=unique,
        increment_canonical_duplicates=len(duplicates),
        increment_rejected=len(rejections),
        increment_rejection_reasons=dict(Counter(r["reason"] for r in rejections)),
        final_unique_full_structures=preserved + unique,
        planned_head_chain_pairs=preserved + len(selection),
        successfully_assembled=preserved + assembled,
        canonical_duplicates=base_report["canonical_duplicates"] + len(duplicates),
        total_rejected=base_report["total_rejected"] + len(rejections),
        within_2000000=preserved + unique <= 2000000,
        baseline_library_sha256=digest,
        no_automatic_refill=True,
        increment_by_chemistry_family=dict(counters["family"]),
        increment_by_structure_origin=dict(counters["origin"]),
        increment_by_plausibility_tier=dict(counters["tier"]),
        increment_by_linkage=dict(counters["linkage"]),
    )
    for field, key in (
        ("full_by_chemistry_family", "family"),
        ("by_structure_origin", "origin"),
        ("by_plausibility_tier", "tier"),
    ):
        report[field] = dict(Counter(base_report[field]) + counters[key])
    reasons = Counter(base_report.get("rejection_reasons", {}))
    reasons.update(r["reason"] for r in rejections)
    report["rejection_reasons"] = dict(reasons)
    report["sanitize_rejected"] = sum(
        reasons[key] for key in ("invalid_valence", "sanitize_failed", "invalid_smiles")
    )
    report["formula_rejected"] = reasons["formula_invalid"]
    for field, key in (("AA", "AA"), ("OA", "OA"), ("PE_PA", "PA"), ("lyso", "lyso")):
        report[field] = base_report[field] + counters["linkage"][key]
    report["family_proportions"] = {
        k: n / (preserved + unique) for k, n in report["full_by_chemistry_family"].items()
    }
    coverage = pq.read_table(library / "polar_coverage.parquet").to_pylist()
    for row in coverage:
        row["final_full_structure_count"] += counters["head"][row["polar_id"]]
    counts = [r["final_full_structure_count"] for r in coverage]
    report.update(
        structures_per_polar_min=min(counts),
        structures_per_polar_median=median(counts),
        structures_per_polar_max=max(counts),
    )
    pq.write_table(
        pa.Table.from_pylist(coverage), output / "polar_coverage.parquet", compression="zstd"
    )
    for name in ("special_cases.parquet",):
        original = pq.read_table(library / name)
        extra = pq.read_table(
            output / "increment_structures.parquet", filters=[("is_special_case", "=", True)]
        )
        pq.write_table(pa.concat_tables([original, extra]), output / name, compression="zstd")
    report["special_head_background_counts"] = {
        head: n + counters["head"][head]
        for head, n in base_report["special_head_background_counts"].items()
    }
    report["special_full_structure_count"] = sum(report["special_head_background_counts"].values())
    old_coverage = read_json(library / "chain_coverage_report.json")
    old_tokens = old_coverage["chain_token_counts"]
    dump_json(
        output / "chain_coverage_report.json",
        chain_coverage(chains, Counter(old_tokens) + counters["token"]),
    )
    report["structures_sha256"] = file_hash(output / "structures.parquet")
    dump_json(output / "library_report.json", report)
    verify_frozen(frozen)
    dump_json(
        output / "preservation_verification.json",
        {
            "status": "passed",
            "frozen_files_checked": len(frozen),
            "all_original_inputs_rehashed": True,
            "original_input_hashes_json_preserved_byte_for_byte": file_hash(
                library / "input_hashes.json"
            )
            == file_hash(output / "input_hashes.json"),
            "all_original_records_and_columns_equal_combined_prefix": preserved,
            "baseline_library_sha256": digest,
            "training_performed": False,
            "dataset_split_created": False,
        },
    )
    return report
