"""Immutable inputs and runtime contract for the new 8,000-template experiment."""

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from omegaconf import OmegaConf

from structure_pretrain.config import ModelConfig
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer

CONFIG = Path("configs/glycerol_core_templates8000_v1.json")
ROOT = Path("outputs/glycerol_core_templates8000_v1_20261026")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def emit(**row):
    print(json.dumps(row, ensure_ascii=False), flush=True)


def configuration():
    c = read(CONFIG)
    assert Path(c["root"]) == ROOT
    assert c["template_pool_size"] == 8000 and c["K"] == 3
    assert c["fp32"] and c["attention"] == "math" and not c["tf32"]
    assert c["checkpoint_selection"] == "minimum_full_validation_token_CE"
    return c


def model_config():
    c = configuration()
    config = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(ModelConfig()),
            OmegaConf.load(c["base_model_configuration"])["model"],
        )
    )
    config.max_polar_length = c["max_label_tokens"]
    return config


def tokenizer(root=ROOT):
    return SmilesTokenizer(read(Path(root) / "data/tokenizer.json")["extra_tokens"])


def verify_data(root=ROOT):
    folder = Path(root) / "data"
    manifest = read(folder / "manifest.json")
    if manifest["status"] != "passed" or manifest["configuration_sha256"] != digest(CONFIG):
        raise ValueError("New core configuration/data preparation changed")
    for name, sha in manifest["assets_sha256"].items():
        if digest(folder / name) != sha:
            raise ValueError("New core data changed: " + name)
    for name, sha in manifest["inputs_sha256"].items():
        if digest(name) != sha:
            raise ValueError("Frozen source changed: " + name)
    return manifest


def source_paths():
    paths = [
        CONFIG,
        Path(configuration()["base_model_configuration"]),
        Path("src/structure_pretrain/config.py"),
        Path("src/structure_pretrain/data/smiles_tokenizer.py"),
        Path("src/structure_pretrain/reported_headgroups/formal_template_data.py"),
        Path("src/structure_pretrain/reported_headgroups/template_selection.py"),
        Path("src/structure_pretrain/formal_v2/retrieval.py"),
        Path("src/structure_pretrain/formal_v2/retention.py"),
        Path("src/structure_pretrain/formal_v2/schedule.py"),
        Path("src/structure_pretrain/formal_v2/common.py"),
        Path("scripts/supervise_formula_beam_cuda_v04.py"),
        Path("tests/test_core_templates.py"),
    ]
    for folder in (
        "core_templates",
        "models",
        "utils",
        "glycerol_core_collection",
        "glycerol_substructure_collection",
        "headgroup_target_expansion",
    ):
        paths.extend(sorted(Path("src/structure_pretrain", folder).glob("*.py")))
    paths.extend(sorted(Path("scripts").glob("*core_templates8000*.py")))
    return list(dict.fromkeys(paths))


def contract(root=ROOT):
    root = Path(root)
    manifest = read(root / "data/manifest.json")
    tests = read(root / "unit_tests.json")
    if not tests["passed"] or tests["source_sha256"] != digest("tests/test_core_templates.py"):
        raise ValueError("Matching new-library leakage/chemistry/runtime tests required")
    if manifest["configuration_sha256"] != digest(CONFIG):
        raise ValueError("Configuration changed since preparation")
    return {
        "schema": configuration()["schema"],
        "configuration_sha256": digest(CONFIG),
        "data_manifest_sha256": digest(root / "data/manifest.json"),
        "source_sha256": {p.as_posix(): digest(p) for p in source_paths()},
        "unit_tests_sha256": digest(root / "unit_tests.json"),
        "model": asdict(model_config()),
        "runtime": {
            "python": platform.python_version(),
            **{n: version(n) for n in ("torch", "rdkit", "numpy", "pyarrow", "usearch")},
        },
    }
