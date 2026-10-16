import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.retrieval.indexed_morgan import (
    IndexedMorganRetriever,
    exact_benchmark,
    full_readiness,
    make_index,
    prepare_cache,
    rank_groups,
)


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
