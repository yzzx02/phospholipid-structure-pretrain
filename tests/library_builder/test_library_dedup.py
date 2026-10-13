import csv
from dataclasses import replace

from structure_pretrain.library_builder.assembly import assemble
from structure_pretrain.library_builder.chain_combinations import make_combination
from structure_pretrain.library_builder.chain_library import import_chains
from structure_pretrain.library_builder.deduplicate import StructureStore
from structure_pretrain.library_builder.polar_library import import_polars

from .conftest import EXAMPLES


def test_canonical_dedup_merges_provenance_and_anchor_flag(inputs, tmp_path):
    chains, polars, _ = inputs
    row = assemble(chains, make_combination(("A-16:0", "A-18:1"), chains), polars["demo_PC"])
    store = StructureStore(tmp_path / "dedup.sqlite", 1)
    assert store.add(row)
    second = dict(row, source_polar_ids=["another_source"], is_anchor_combination=True)
    assert not store.add(second)
    merged = list(store.records())[0]
    assert store.count == 1 and store.duplicates_removed == 1
    assert merged["source_polar_ids"] == ["another_source", "demo_PC"]
    assert merged["is_anchor_combination"] and merged["structure_id"].startswith("pl_")
    store.close()


def test_duplicate_polar_sources_merge(inputs, tmp_path):
    _, polars, _ = inputs
    first = polars["demo_PC"]
    rows = [first.row(), replace(first, polar_id="another_PC", source_id="other_origin").row()]
    path = tmp_path / "polars.tsv"
    fields = [name for name in rows[0] if name not in ("source_polar_ids", "provenance")]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    errors = []
    imported = import_polars(path, errors)
    assert len(imported) == 1 and not errors
    assert set(imported[0].source_polar_ids) == {"demo_PC", "another_PC"}


def test_conflicting_chain_representatives_excluded(tmp_path):
    with (EXAMPLES / "chains.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows, fields = list(reader), reader.fieldnames
    original = next(row for row in rows if row["chain_token"] == "O-16:1")
    conflict = dict(
        original,
        representative_fragment_smiles="[*:1]CCCCCCCC=CCCCCCCC",
        representative_double_bond_positions="[8]",
    )
    # This alternate explicit source rule has the same token but a different double-bond position.
    rows.append(conflict)
    path = tmp_path / "chains.tsv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    errors = []
    imported = import_chains(path, errors)
    assert "O-16:1" not in imported
    assert any(error.reason == "conflicting_chain_representatives" for error in errors)


def test_mixed_p_annotation_conventions_do_not_create_ambiguous_labels(tmp_path):
    with (EXAMPLES / "chains.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows, fields = list(reader), reader.fieldnames
    p = next(row for row in rows if row["chain_token"] == "P-16:0")
    rows.append(
        dict(p, chain_token="P-16:1", double_bond_count="1", p_db_count_includes_vinyl="true")
    )
    path = tmp_path / "mixed.tsv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    errors = []
    chains = import_chains(path, errors)
    assert all(chain.linkage_type != "P" for chain in chains.values())
    assert sum(error.reason == "conflicting_p_db_conventions" for error in errors) == 2
