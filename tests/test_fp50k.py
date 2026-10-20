"""Bounded FP50k data order and immutable destination checks."""

import importlib.util
from pathlib import Path

import pytest

from structure_pretrain.polar_only_v04.fp50k_data import fresh_directory


def sampler_class():
    spec = importlib.util.spec_from_file_location("real_trainer", "train_structure_real.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReproducibleBatchSampler


def test_resume_across_pool_epoch_boundary():
    sampler = sampler_class()
    full = list(sampler(50, 16, 8, 42))
    assert full[3:] == list(sampler(50, 16, 5, 42, consumed=48))
    assert len({x for row in full[:3] for x in row}) == 48
    assert full != list(sampler(50, 16, 8, 43))


def test_output_cannot_escape_or_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("outputs/independent")
    fresh_directory(root / "new", root)
    with pytest.raises(FileExistsError):
        fresh_directory(root / "new", root)
    with pytest.raises(ValueError):
        fresh_directory(Path("outputs/history"), root)
    with pytest.raises(ValueError):
        fresh_directory(Path("private/experiment"), Path("private"))


def graph_analysis():
    spec = importlib.util.spec_from_file_location(
        "fp50k_analysis", "scripts/analyze_polar_fp50k.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_graph_diagnostics_ignore_smiles_order_and_detect_isomers():
    from structure_pretrain.data.prepare_v03 import atom_counts
    from structure_pretrain.library_builder.canonicalize import parse_molecule

    analysis = graph_analysis()
    a, reordered = "COP(=O)(OCC)[*:1]", "CCOP(=O)(OC)[*:1]"
    b = "O=P(OCCC)(O)[*:1]"
    assert analysis.graph_info(a)[1] == analysis.graph_info(reordered)[1]
    assert analysis.similarity(a, reordered) == 1.0
    assert (atom_counts(parse_molecule(a)) == atom_counts(parse_molecule(b))).all()
    assert analysis.graph_info(a)[1] != analysis.graph_info(b)[1]
    assert analysis.similarity(a, b) < 1.0


def test_graph_diagnostics_follow_attachment_in_multi_p():
    analysis = graph_analysis()
    a = "OP(=O)(O)OP(=O)(OCC)[*:1]"
    b = "[*:1]P(=O)(O)OP(=O)(O)OCC"
    assert (
        analysis.graph_info(a)[1]["phosphate_neighborhood"]
        != analysis.graph_info(b)[1]["phosphate_neighborhood"]
    )
