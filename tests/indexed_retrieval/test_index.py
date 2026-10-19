import hashlib
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.retrieval.indexed_morgan import (
    IndexedMorganRetriever,
    exact_benchmark,
    exact_zero_check,
    full_readiness,
    make_index,
    prepare_cache,
    rank_groups,
    selected_parquet_rows,
)


def test_sparse_parquet_selection_skips_empty_batches_and_preserves_absolute_rows(tmp_path):
    source = tmp_path / "rows.parquet"
    pq.write_table(
        pa.table({"id": list(range(10)), "text": [f"row-{i}" for i in range(10)]}), source
    )
    assert list(selected_parquet_rows(source, [9, 0, 9], ["text"], batch_size=2)) == [
        {"text": "row-0"},
        {"text": "row-9"},
    ]
    assert list(selected_parquet_rows(source, [], ["text"], batch_size=2)) == []
    import pytest

    with pytest.raises(ValueError, match="outside"):
        list(selected_parquet_rows(source, [10], ["text"]))


def test_native_tanimoto_and_rdkit_exact_scores_agree():
    vectors = np.zeros((3, 256), dtype=np.uint8)
    vectors[:, 0] = (1, 3, 7)
    index = make_index(vectors)
    matches = index.search(vectors[0], count=3)
    rows = {0: [0], 1: [1], 2: [2]}
    ids = np.array(["a", "b", "c"], dtype="S67")
    hits = rank_groups(
        vectors[0], matches.keys, vectors, lambda g, count: rows[g][:count], ids, 0, threshold=0.0
    )
    assert hits == [(1, 0.5), (2, 1 / 3)]


def test_canonical_targets_excluded_but_identical_fps_retained(tmp_path):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    # Long saturated chains deliberately produce many identical radius-2 fingerprints.
    smiles = ["C" * n for n in range(9, 35)] + ["C" * n + "O" for n in range(9, 35)]
    records = []
    for i, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        records.append(
            {
                "structure_id": "pl_" + hashlib.sha256(canonical.encode()).hexdigest(),
                "morgan_fp_2048": DataStructs.BitVectToBinaryText(generator.GetFingerprint(mol)),
                "chemistry_family": "phosphate",
                "polar_plausibility_tier": "CORE",
                "structure_origin": "pubchem_baseline",
            }
        )
    source = tmp_path / "structures.parquet"
    pq.write_table(pa.Table.from_pylist(records), source)
    cache = prepare_cache(source, tmp_path / "cache")
    assert len(cache.vectors) < len(cache.fp)
    benchmark = exact_benchmark(cache, tmp_path / "cache", target_count=20, reference_count=40)
    assert benchmark["status"] == "passed"
    assert benchmark["selected_settings"]["tie_aware_recall_at_10"] == 1.0
    report = full_readiness(cache, benchmark, tmp_path / "cache")
    assert report["overall"]["ten_fraction"] == 1.0
    assert report["overall"]["max_similarity"]["mean"] == 1.0
    retriever = IndexedMorganRetriever(tmp_path / "cache")
    hits = retriever.retrieve_smiles(smiles[0])
    assert len(hits) == 10 and all(h["structure_similarity"] == 1.0 for h in hits)
    assert records[0]["structure_id"] not in {h["structure_id"] for h in hits}
    assert len({h["structure_id"] for h in hits}) == len(hits)
    # A cached source may be reused only while its full-library file hash is unchanged.
    assert prepare_cache(source, tmp_path / "cache").manifest == cache.manifest


def test_exact_rerank_filters_subthreshold_and_deduplicates_group_keys():
    vectors = np.zeros((2, 256), dtype=np.uint8)
    vectors[:, 0] = (1, 255)
    ids = np.array(["a", "b", "c"], dtype="S67")
    postings = {0: [0, 1], 1: [2]}
    hits = rank_groups(vectors[0], [0, 0, 1], vectors, lambda g, count: postings[g][:count], ids, 0)
    assert hits == [(1, 1.0)]


def test_zero_template_exact_confirmation_recovers_missed_graphs():
    vectors = np.zeros((3, 256), dtype=np.uint8)
    vectors[:, 0] = (1, 2, 3)
    rows = {0: [0, 1], 1: [2], 2: [3]}
    cache = SimpleNamespace(
        vectors=vectors,
        fp=vectors[[0, 0, 1, 2]],
        offsets=np.array([0, 2, 3, 4]),
        ids=np.array(["a", "b", "c", "d"], dtype="S67"),
        members=lambda g, n: rows[g][:n],
        rdkit_group=lambda g: DataStructs.CreateFromBinaryText(bytes(vectors[g])),
    )
    counts, maximum, means = np.array([0, 1, 1]), np.zeros(3), np.zeros(3)
    report = exact_zero_check(cache, counts, maximum, means)
    assert report["groups_corrected"] == 1
    assert counts[0] == 2 and maximum[0] == 1 and means[0] == 0.75
