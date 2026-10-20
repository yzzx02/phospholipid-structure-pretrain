import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_preserves_previous_raw_provenance_and_adds_source(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = load("import_reported_headgroups_v01")
    library, output = tmp_path / "old", tmp_path / "new"
    (library / "raw").mkdir(parents=True)
    old = {"source": "fixture", "source_id_or_doi": "old"}
    old_bytes = (json.dumps(old) + "\n").encode()
    (library / "raw/user_imports.jsonl").write_bytes(old_bytes)
    record = {
        "full_name": "test phosphoethanolamine",
        "original_smiles": "NCCOP(=O)(O)[*:1]",
        "source": "fixture",
        "source_id_or_doi": "new",
        "source_url": "https://example.org/test",
        "source_kind": "test_fixture_not_research_data",
        "structure_input_kind": "standard_polar",
        "aliases": '["PE fixture", "test alias"]',
    }
    incoming = tmp_path / "input.jsonl"
    incoming.write_text(json.dumps(record) + "\n")
    calls = []
    monkeypatch.setattr(module, "collect", lambda root: calls.append(root))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import",
            "--input",
            str(incoming),
            "--library-root",
            str(library),
            "--output-root",
            str(output),
        ],
    )
    module.main()
    assert (library / "raw/user_imports.jsonl").read_bytes() == old_bytes
    merged = module.read_jsonl(output / "raw/user_imports.jsonl")
    assert len(merged) == 2 and merged[0] == old
    assert merged[1]["literature_verified"] is False
    assert merged[1]["aliases"] == ["PE fixture", "test alias"]
    assert calls == [output]
    with pytest.raises(ValueError, match="never overwritten"):
        module.main()


def test_import_requires_source_and_explicit_input_boundary(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = load("import_reported_headgroups_v01")
    incoming = tmp_path / "missing.jsonl"
    incoming.write_text(json.dumps({"full_name": "PC"}) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import",
            "--input",
            str(incoming),
            "--library-root",
            str(tmp_path / "old"),
            "--output-root",
            str(tmp_path / "new"),
        ],
    )
    with pytest.raises(ValueError, match="Required nonempty fields"):
        module.main()
    assert not (tmp_path / "new").exists()


def test_query_returns_all_name_collisions_and_reverse_sources(tmp_path):
    module = load("query_reported_headgroups_v01")

    def write(name, records):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write("reported_headgroups_v01.csv", [{"headgroup_id": "HG-A"}, {"headgroup_id": "HG-B"}])
    write(
        "headgroup_name_alias_mapping.csv",
        [
            {"query_name": "PPA", "normalized_name": "ppa", "headgroup_id": "HG-A"},
            {"query_name": "PPA", "normalized_name": "ppa", "headgroup_id": "HG-B"},
        ],
    )
    write(
        "headgroup_sources.csv",
        [{"headgroup_id": "HG-A", "source": "first"}, {"headgroup_id": "HG-B", "source": "second"}],
    )
    assert len(module.query(tmp_path, name=" pPa ")) == 2
    assert module.query(tmp_path, identifier="HG-B")[0]["all_sources"][0]["source"] == "second"
    assert module.query(tmp_path, name="missing") == []
