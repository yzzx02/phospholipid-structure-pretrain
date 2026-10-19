"""Build a separate Morgan cache/index and verify v0.3 queries, without training."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from structure_pretrain.library_builder.incremental import read_json
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_plausibility.pipeline import dump_json
from structure_pretrain.retrieval.indexed_morgan import (
    IndexedMorganRetriever,
    exact_benchmark,
    make_index,
    prepare_cache,
)


def build(library, output, threads):
    library, output = Path(library), Path(output)
    audit = read_json(library / "verification_audit.json")
    if audit["status"] != "passed" or audit["structures_sha256"] != file_hash(
        library / "structures.parquet"
    ):
        raise ValueError("Independent increment/preservation audit must pass before indexing")
    cache = prepare_cache(library / "structures.parquet", output)
    benchmark_path = output / "exact_benchmark.json"
    benchmark = (
        read_json(benchmark_path) if benchmark_path.exists() else exact_benchmark(cache, output)
    )
    if benchmark["status"] != "passed":
        raise ValueError("Index recall benchmark failed")
    path = output / "morgan_groups.usearch"
    if not path.exists():
        start = time.monotonic()
        index = make_index(
            cache.vectors,
            expansion_search=benchmark["selected_settings"]["expansion_search"],
            threads=threads,
        )
        index.save(path.with_suffix(".partial"))
        path.with_suffix(".partial").replace(path)
        del index
        print(f"v0.3 index built in {time.monotonic() - start:.1f}s", flush=True)
    retriever = IndexedMorganRetriever(output)
    rng = np.random.default_rng(107)
    sample = set(rng.choice(len(cache.fp), 100, replace=False).tolist())
    source = pq.ParquetFile(library / "increment_structures.parquet")
    increment_samples = (
        source.read(columns=["canonical_smiles", "structure_id"])
        .take(
            pa_indices := sorted(
                rng.choice(
                    source.metadata.num_rows, min(100, source.metadata.num_rows), replace=False
                ).tolist()
            )
        )
        .to_pylist()
    )
    assert len(pa_indices) == len(increment_samples)
    checked = 0

    def check(row):
        nonlocal checked
        hits = retriever.retrieve_smiles(row["canonical_smiles"])
        assert len(hits) > 0
        assert row["structure_id"] not in {h["structure_id"] for h in hits}
        assert len({h["structure_id"] for h in hits}) == len(hits)
        assert all(h["structure_similarity"] >= 0.4 for h in hits)
        checked += 1

    for row in increment_samples:
        check(row)
    offset = 0
    for batch in pq.ParquetFile(library / "structures.parquet").iter_batches(
        batch_size=10000, columns=["canonical_smiles", "structure_id"]
    ):
        positions = sorted(i - offset for i in sample if offset <= i < offset + batch.num_rows)
        for row in batch.take(positions).to_pylist():
            check(row)
        offset += batch.num_rows
    result = {
        "status": "passed",
        "version": "v0.3",
        "row_count": len(cache.fp),
        "unique_fingerprint_count": len(cache.vectors),
        "query_checks": checked,
        "increment_queries": len(increment_samples),
        "structures_sha256": cache.manifest["structures_sha256"],
        "index_sha256": file_hash(path),
        "fingerprint": {"radius": 2, "bits": 2048, "includeChirality": False},
        "exact_benchmark": benchmark["selected_settings"],
        "full_v03_retrieval_census_performed": False,
        "training_performed": False,
        "dataset_split_created": False,
    }
    dump_json(output / "index_verification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", default="data/structure_library_v0.3")
    p.add_argument("--output", default="data/private/morgan_index_v0.3")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    print(json.dumps(build(a.library, a.output, a.threads), indent=2), flush=True)
