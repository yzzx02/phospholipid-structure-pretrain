"""CPU-only exact-score audit of historical HNSW choices; no cache mutation."""

import gzip
import json

import numpy as np
import pyarrow.parquet as pq

from structure_pretrain.formal_v2.common import ROOT, read, write
from structure_pretrain.reported_headgroups.formal_template_data import FEATURES, V03


def main():
    bank = np.load(ROOT / "data/bank_packed.npy")
    fp = np.load(FEATURES / "chainless_fingerprints.npy", mmap_mode="r")
    splits = np.load(ROOT / "data/splits.npy")
    groups = np.flatnonzero(splits != 255)
    lookup = np.empty((len(splits), 2), dtype=np.int32)
    lookup[
        np.load(FEATURES / "variant_heads.npy"), 2 - np.load(FEATURES / "variant_occupancy.npy")
    ] = np.arange(len(fp))
    old = np.load("outputs/template_guided_fp5000_20261022_r1/data/references.npy", mmap_mode="r")
    pop = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)
    summary = {}
    for occupancy in (0, 1):
        exact = np.load(ROOT / f"audit/chainless_{occupancy}_scores.npy")[:, 0]
        identities = np.load(ROOT / f"audit/chainless_{occupancy}_indices.npy")[:, 0]
        original = old[groups, occupancy, 0]
        assert np.all(original >= 0)
        similarities = np.zeros(len(groups))
        for start in range(0, len(groups), 4096):
            q = np.asarray(fp[lookup[groups[start : start + 4096], occupancy]])
            b = bank[occupancy, original[start : start + 4096]]
            inter = pop[np.bitwise_and(q, b)].sum(-1)
            union = pop[np.bitwise_or(q, b)].sum(-1)
            similarities[start : start + len(q)] = inter / union
        gap = exact - similarities
        assert not np.any(gap < -1e-6)
        different = identities != original
        summary[str(occupancy)] = {
            "N": len(groups),
            "different_top1_identity": int(different.sum()),
            "strictly_lower_old_score_gt1e6": int((gap > 1e-6).sum()),
            "different_identity_with_tied_score": int((different & (np.abs(gap) <= 1e-6)).sum()),
            "maximum_exact_score_gain": float(gap.max()),
        }
    write(ROOT / "audit/old_HNSW_exact_score_comparison.json", summary)
    heads = (
        pq.read_table(V03 / "polar_groups.parquet", columns=["canonical_polar_smiles"])
        .column(0)
        .to_pylist()
    )
    identities = read(ROOT / "data/template_identity_private.json")
    pscore = np.load(ROOT / "audit/polar_scores.npy")[:, 0]
    pidx = np.load(ROOT / "audit/polar_indices.npy")[:, 0]
    with gzip.open(
        ROOT / "audit/fingerprint_disagreement_cases_private.jsonl.gz", "wt", encoding="utf-8"
    ) as stream:
        for occupancy in (0, 1):
            cscore = np.load(ROOT / f"audit/chainless_{occupancy}_scores.npy")[:, 0]
            cidx = np.load(ROOT / f"audit/chainless_{occupancy}_indices.npy")[:, 0]
            options = np.flatnonzero(pidx != cidx)
            chosen = options[np.argsort(np.abs(cscore[options] - pscore[options]))[-5:]]
            for i in chosen:
                stream.write(
                    json.dumps(
                        {
                            "group": int(groups[i]),
                            "chain_count": 2 - occupancy,
                            "query_polar": heads[int(groups[i])],
                            "Polar_top1": identities[int(pidx[i])],
                            "chainless_top1": identities[int(cidx[i])],
                            "Polar_score": float(pscore[i]),
                            "chainless_score": float(cscore[i]),
                        }
                    )
                    + "\n"
                )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
