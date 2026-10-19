"""Metadata-only held-out head similarity/scaffold difficulty; never model selection."""

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.polar_plausibility.pipeline import dump_json
from structure_pretrain.retrieval.indexed_morgan import distribution


def audit(root, sample_count=256):
    root = Path(root)
    started = time.monotonic()
    groups = pq.read_table(root / "polar_groups.parquet").to_pylist()
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    train = [g for g in groups if g["split"] == "train"]
    fps = [generator.GetFingerprint(parse_molecule(g["canonical_polar_smiles"])) for g in train]
    rng = np.random.default_rng(42)
    results = {}
    for name in ("validation", "test"):
        held = [g for g in groups if g["split"] == name]
        selected = rng.choice(len(held), min(sample_count, len(held)), replace=False)
        values = []
        for i in selected:
            query = generator.GetFingerprint(parse_molecule(held[i]["canonical_polar_smiles"]))
            values.append(max(DataStructs.BulkTanimotoSimilarity(query, fps)))
        results[name] = {
            "sampled_heads": len(selected),
            "total_heads": len(held),
            "top1_head_similarity_to_train": distribution(values),
            "fraction_at_least_0.9": float(np.mean(np.array(values) >= 0.9)),
            "fingerprint_one_is_not_identical_graph": int(np.count_nonzero(np.array(values) == 1)),
        }
    result = {
        "schema": "pretraining-v03-head-generalization-difficulty",
        "structures_sha256": read_json(root / "labels_manifest.json")["structures_sha256"],
        "status": "complete",
        "seed": 42,
        "train_heads": len(train),
        "scope": "exact against all train heads, sampled held-out head metadata; "
        "no model/test tuning",
        "fingerprint": {"radius": 2, "fpSize": 2048, "includeChirality": False},
        "results": results,
        "seconds": time.monotonic() - started,
        "interpretation": "Near analogs and shared scaffolds concern difficulty; "
        "exact graph leakage is checked separately",
    }
    dump_json(root / "head_difficulty_audit.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="data/private/pretraining_v03")
    a = p.parse_args()
    audit(a.prepared)
