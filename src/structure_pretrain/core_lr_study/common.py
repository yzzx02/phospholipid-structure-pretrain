"""Frozen inputs and contracts without changing either completed experiment."""

import hashlib
from pathlib import Path

from structure_pretrain.core_templates.common import (
    ROOT as BASE_ROOT,
)
from structure_pretrain.core_templates.common import (
    contract as base_contract,
)
from structure_pretrain.core_templates.common import (
    digest,
    model_config,
    read,
    write,
)
from structure_pretrain.core_templates.common import (
    verify_data as verify_base_data,
)

CONFIG = Path("configs/core_lr_schedule_study_20261028.json")
ROOT = Path("outputs/core_lr_schedule_study_20261028")


def configuration():
    c = read(CONFIG)
    assert Path(c["root"]) == ROOT and Path(c["base_root"]) == BASE_ROOT
    assert c["minimum_epochs"] == 50 and c["max_epochs"] == c["planned_schedule_epochs"] == 80
    assert c["schedule_total_steps"] == c["steps_per_epoch"] * 80 == 427200
    assert c["batch_size"] == 32 and c["seed"] == 42 and c["K"] == 3
    assert c["fp32"] and c["attention"] == "math" and not c["tf32"]
    return c


def original_run():
    return BASE_ROOT / "train_main_k3_seed42"


def parameter_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.named_parameters():
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_historical():
    current = base_contract()
    assert current == read(BASE_ROOT / "queue_started.json")["contract"]
    from structure_pretrain.core_ablation.common import ROOT as CONTROL_ROOT
    from structure_pretrain.core_ablation.common import contract as control_contract

    assert control_contract() == read(CONTROL_ROOT / "queue_started.json")["contract"]
    return current


def verify_data(root=ROOT):
    root, c = Path(root), configuration()
    m = read(root / "data/manifest.json")
    assert m["status"] == "passed" and m["configuration_sha256"] == digest(CONFIG)
    for name, sha in m["assets_sha256"].items():
        assert digest(root / "data" / name) == sha, name
    verify_base_data(BASE_ROOT)
    assert digest(read(original_run() / "last.json")["path"]) == c["original_E3_epoch48_sha256"]
    assert digest(read(original_run() / "best.json")["path"]) == c["original_E3_best_sha256"]
    return m


def source_paths():
    files = [
        CONFIG,
        Path("tests/test_core_lr_study.py"),
        Path("scripts/supervise_formula_beam_cuda_v04.py"),
    ]
    files.extend(sorted(Path("src/structure_pretrain/core_lr_study").glob("*.py")))
    files.extend(
        sorted(
            p
            for p in Path("scripts").glob("*core_lr_study*.py")
            if not p.name.startswith("report_")
        )
    )
    return files


def contract(root=ROOT):
    root = Path(root)
    base = verify_historical()
    unit = read(root / "unit_tests.json")
    assert unit["passed"] and unit["source_sha256"] == digest("tests/test_core_lr_study.py")
    return {
        "schema": configuration()["schema"],
        "configuration_sha256": digest(CONFIG),
        "data_manifest_sha256": digest(root / "data/manifest.json"),
        "unit_tests_sha256": digest(root / "unit_tests.json"),
        "source_sha256": {p.as_posix(): digest(p) for p in source_paths()},
        "historical_template_contract": base,
        "historical_formula_control_contract_sha256": digest(
            "outputs/core_formula_comparison_20261027/queue_started.json"
        ),
    }


__all__ = [
    "CONFIG",
    "ROOT",
    "BASE_ROOT",
    "configuration",
    "contract",
    "digest",
    "model_config",
    "original_run",
    "parameter_digest",
    "read",
    "verify_data",
    "verify_historical",
    "write",
]
