import pyarrow.parquet as pq

from structure_pretrain.library_builder.pipeline import build_library

from .conftest import EXAMPLES


def test_all_heads_keep_common_anchors(inputs, tmp_path):
    _, polars, config = inputs
    report = build_library(
        config,
        EXAMPLES / "chains.tsv",
        EXAMPLES / "polar_moieties.tsv",
        tmp_path / "anchors",
        EXAMPLES / "p_to_o_mapping.tsv",
    )
    assert report["mandatory_anchor_count"] == 28
    assert set(report["anchor_combo_counts"].values()) == {4}
    rows = pq.read_table(tmp_path / "anchors" / "structures.parquet").to_pylist()
    for head in polars:
        combos = {
            row["combination_id"]
            for row in rows
            if row["polar_id"] == head and row["is_anchor_combination"]
        }
        assert combos == set(config.anchor_combinations)


def test_missing_anchor_is_reported(inputs):
    from structure_pretrain.library_builder.chain_combinations import active_anchors

    chains, _, config = inputs
    del chains["A-22:6"]
    errors = []
    assert len(active_anchors(chains, config, errors)) == 6
    assert errors[0].reason == "anchor_chain_missing"
