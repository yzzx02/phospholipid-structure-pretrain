"""Formal V2 configuration and immutable shared-input helpers."""

import hashlib
import json
from pathlib import Path

from structure_pretrain.reported_headgroups.formal_template_data import digest, read, write

CONFIG = Path("configs/template_guided_v2.json")
ROOT = Path("outputs/template_guided_v2_20261023")


def configuration():
    value = read(CONFIG)
    assert Path(value["root"]) == ROOT
    assert value["batch_size"] == 32 and value["K"] == 3 and value["max_epochs"] == 80
    assert value["steps_per_epoch"] * value["max_epochs"] == value["max_optimizer_steps"] == 428720
    assert value["fp32"] and value["attention"] == "math" and not value["tf32"]
    return value


def emit(**record):
    print(json.dumps(record, ensure_ascii=False), flush=True)


def array_digest(array):
    """Bind ordered query/bank bytes, including dtype and shape, without a copy."""
    import numpy as np

    array = np.ascontiguousarray(array)
    h = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def verify_data(root=ROOT):
    root = Path(root)
    manifest = read(root / "data/manifest.json")
    if manifest["configuration_sha256"] != digest(CONFIG):
        raise ValueError("Frozen V2 configuration changed")
    for name, expected in manifest["data_assets_sha256"].items():
        if digest(root / "data" / name) != expected:
            raise ValueError("V2 data hash changed: " + name)
    for path, expected in manifest["frozen_input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Original immutable input changed: " + path)
    return manifest


def contract(root=ROOT):
    import platform
    from dataclasses import asdict
    from importlib.metadata import version

    from structure_pretrain.reported_headgroups.formal_template_data import model_config

    root = Path(root)
    for phase in ("A", "B", "C"):
        proof = read(root / "audit" / f"{phase}_passed.json")
        if proof["status"] != "passed":
            raise ValueError("Pretraining prerequisite failed: " + phase)
    a = read(root / "audit/A_passed.json")
    if a["data_manifest_sha256"] != digest(root / "data/manifest.json"):
        raise ValueError("Audited data manifest changed")
    b = read(root / "audit/B_passed.json")
    if b["visual_QA"] != "passed":
        raise ValueError("PDF render/visual QA required")
    for name, expected in b["assets_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Coverage report/figure changed: " + name)
    unit = read(root / "unit_tests.json")
    if not unit["passed"] or unit["test_source_sha256"] != digest("tests/test_formal_v2.py"):
        raise ValueError("Matching failure-denominator/schedule/retention tests required")

    paths = [
        CONFIG,
        Path(configuration()["base_model_configuration"]),
        Path("configs/template_guided_fp5000.json"),
        Path("scripts/prepare_template_guided_v2.py"),
        Path("scripts/audit_template_guided_v2_coverage.py"),
        Path("scripts/check_template_guided_v2_generation.py"),
        Path("scripts/gate_template_guided_v2.py"),
        Path("scripts/supervise_template_guided_v2.py"),
        Path("scripts/run_template_guided_v2.py"),
        Path("scripts/evaluate_template_guided_v2.py"),
        Path("scripts/plot_template_guided_v2.py"),
        Path("scripts/train_template_guided_v2.py"),
        Path("train_template_guided_fp5000.py"),
        Path("src/structure_pretrain/reported_headgroups/formal_template_data.py"),
        Path("src/structure_pretrain/reported_headgroups/template_selection.py"),
        Path("src/structure_pretrain/training/quality_v03.py"),
        Path("src/structure_pretrain/data/smiles_tokenizer.py"),
        Path("src/structure_pretrain/utils/attention.py"),
        Path("src/structure_pretrain/utils/formula.py"),
        Path("src/structure_pretrain/utils/masks.py"),
        Path("tests/test_formal_v2.py"),
    ]
    for folder in ("formal_v2", "models", "polar_only_v04", "library_builder"):
        paths.extend(sorted(Path("src/structure_pretrain", folder).glob("*.py")))
    return {
        "schema": configuration()["schema"],
        "configuration_sha256": digest(CONFIG),
        "data_manifest_sha256": digest(Path(root) / "data/manifest.json"),
        "source_sha256": {p.as_posix(): digest(p) for p in paths},
        "model": asdict(model_config()),
        "runtime_versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "torch",
                    "torch-geometric",
                    "numpy",
                    "rdkit",
                    "scipy",
                    "usearch",
                    "omegaconf",
                )
            },
        },
        "prerequisites": {
            phase: digest(Path(root) / "audit" / f"{phase}_passed.json")
            for phase in ("A", "B", "C")
        },
        "unit_tests_sha256": digest(root / "unit_tests.json"),
        "audit_report_sha256": {
            name: digest(root / "audit" / (name + ".json"))
            for name in (
                "A1_integrity",
                "A2_dual_fingerprint",
                "A3_retrieval",
                "B1_known_references",
                "B2_coverage",
                "cluster_diagnostic",
            )
        },
    }


def dedicated_root():
    root = ROOT.resolve()
    assert root.is_relative_to(Path("outputs").resolve())
    root.mkdir(exist_ok=True)
    return root


__all__ = ["CONFIG", "ROOT", "configuration", "dedicated_root", "digest", "emit", "read", "write"]
