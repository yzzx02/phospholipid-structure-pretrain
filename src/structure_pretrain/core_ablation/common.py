"""Independent contract; preserve both historical source trees and all checkpoints."""

from dataclasses import asdict
from pathlib import Path

from structure_pretrain.core_templates.common import (
    ROOT as TEMPLATE_ROOT,
)
from structure_pretrain.core_templates.common import (
    contract as template_contract,
)
from structure_pretrain.core_templates.common import (
    digest,
    model_config,
    read,
    tokenizer,
    write,
)
from structure_pretrain.core_templates.common import (
    verify_data as verify_template_data,
)

CONFIG = Path("configs/core_formula_comparison_20261027.json")
ROOT = Path("outputs/core_formula_comparison_20261027")


def configuration():
    c = read(CONFIG)
    assert Path(c["root"]) == ROOT
    assert c["batch_size"] == 32 and c["beam_size"] == 20 and c["max_label_tokens"] == 96
    assert c["max_optimizer_steps"] == c["steps_per_epoch"] * c["max_epochs"] == 427200
    assert c["fp32"] and c["attention"] == "math" and not c["tf32"]
    return c


def verify_historical():
    baseline = read(TEMPLATE_ROOT / "queue_started.json")["contract"]
    if template_contract() != baseline:
        raise ValueError("Historical 8000-template model source/config contract changed")
    old = read("outputs/template_guided_v2_20261023/queue_started.json")["contract"][
        "source_sha256"
    ]
    if len(old) != 73 or any(digest(path) != sha for path, sha in old.items()):
        raise ValueError("Historical V2 source changed")
    return baseline


def verify_data(root=ROOT):
    root = Path(root)
    manifest = read(root / "data/manifest.json")
    assert manifest["status"] == "passed" and manifest["configuration_sha256"] == digest(CONFIG)
    for name, sha in manifest["assets_sha256"].items():
        if digest(root / "data" / name) != sha:
            raise ValueError("Control input changed: " + name)
    verify_template_data(TEMPLATE_ROOT)
    c = configuration()
    if digest(c["template_checkpoint"]) != c["template_checkpoint_sha256"]:
        raise ValueError("Frozen epoch43 checkpoint changed")
    return manifest


def source_paths():
    paths = [CONFIG, Path("tests/test_core_ablation.py")]
    paths.extend(sorted(Path("src/structure_pretrain/core_ablation").glob("*.py")))
    paths.extend(
        sorted(
            p
            for p in Path("scripts").glob("*core_ablation*.py")
            if not p.name.startswith("report_")
        )
    )
    paths.append(Path("scripts/supervise_formula_beam_cuda_v04.py"))
    paths.append(Path("src/structure_pretrain/polar_only_v04/constrained_decode.py"))
    return paths


def contract(root=ROOT):
    root = Path(root)
    old = verify_historical()
    unit = read(root / "unit_tests.json")
    if not unit["passed"] or unit["source_sha256"] != digest("tests/test_core_ablation.py"):
        raise ValueError("Matching constraints/formula-only tests required")
    return {
        "schema": configuration()["schema"],
        "configuration_sha256": digest(CONFIG),
        "data_manifest_sha256": digest(root / "data/manifest.json"),
        "source_sha256": {p.as_posix(): digest(p) for p in source_paths()},
        "historical_template_contract": old,
        "unit_tests_sha256": digest(root / "unit_tests.json"),
        "model": asdict(model_config()),
    }


__all__ = [
    "CONFIG",
    "ROOT",
    "TEMPLATE_ROOT",
    "configuration",
    "contract",
    "digest",
    "model_config",
    "read",
    "tokenizer",
    "verify_data",
    "verify_historical",
    "write",
]
