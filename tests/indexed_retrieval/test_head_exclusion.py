import hashlib
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from structure_pretrain.retrieval.head_exclusion import (
    exact_excluded,
    full_head_audit,
    paired_group,
    same_head_summary,
)
from structure_pretrain.retrieval.indexed_morgan import (
    exact_benchmark,
    full_readiness,
    prepare_cache,
)


def mock_cache(vectors, postings):
    members = np.array([r for group in range(len(vectors)) for r in postings[group]])
    offsets = np.r_[0, np.cumsum([len(postings[g]) for g in range(len(vectors))])]
    group_of_row = np.zeros(len(members), dtype=np.uint32)
    for group, rows in postings.items():
        group_of_row[rows] = group
    return SimpleNamespace(
        vectors=vectors,
        fp=vectors[group_of_row],
        ids=np.array([f"row_{r:04d}" for r in range(len(members))], dtype="S67"),
        group_of_row=group_of_row,
        offsets=offsets,
        members=lambda g, n: members[offsets[g] : min(offsets[g + 1], offsets[g] + n)],
        rdkit_group=lambda g: DataStructs.CreateFromBinaryText(bytes(vectors[g])),
    )


def test_filter_beyond_posting_prefix_and_fp_identical_different_heads():
    vectors = np.zeros((2, 256), dtype=np.uint8)
    vectors[:, 0] = [3, 7]
    cache = mock_cache(vectors, {0: list(range(34)), 1: list(range(34, 45))})
    codes = np.array([0] * 31 + [1] * 3 + [2] * 11)
    base, excluded = paired_group(cache, codes, 0, [0, 1], [0.0, 1 / 3], max_per_head=31)
    assert len(base[0]) == 10 and all(r != 0 for r, _ in base[0])
    assert len(excluded[0]) == 10
    assert [r for r, _ in excluded[0]][:3] == [31, 32, 33]
    assert all(codes[r] != 0 for r, _ in excluded[0])
    assert all(s == 1 for _, s in excluded[1])
    assert exact_excluded(cache, codes, 0, 0, 31, chunk_size=1) == excluded[0]


def test_same_head_can_occupy_more_than_sixteen_nearest_groups():
    vectors = np.zeros((40, 256), dtype=np.uint8)
    vectors[:, :4] = 255
    for group in range(40):
        vectors[group, 5 + group // 8] |= 1 << (group % 8)
    cache = mock_cache(vectors, {g: [g] for g in range(40)})
    codes = np.array([0] * 22 + [1] * 18)
    _, excluded = paired_group(cache, codes, 0, np.arange(40), np.r_[0.0, np.full(39, 2 / 34)], 22)
    assert len(excluded[0]) == 10
    assert all(row >= 22 for row, _ in excluded[0])
    assert all(abs(score - 32 / 34) < 1e-12 for _, score in excluded[0])


def test_same_head_pair_and_target_denominators_with_missing_neighbors():
    hits = np.array([[1, 2, -1], [0, -1, -1], [0, 1, -1]])
    codes = np.array([7, 7, 8])
    result = same_head_summary(hits, np.zeros_like(hits), codes)
    assert result["same_polar_id_pair_fraction"] == 2 / 5
    assert result["targets_with_any_same_polar_id_fraction"] == 2 / 3
    assert result["top1_same_polar_id_fraction_among_usable"] == 2 / 3


@pytest.mark.parametrize("head_counts", [(16, 16, 16), (20, 3)])
def test_paired_full_census_resume_and_rare_exhaustive_confirmation(tmp_path, head_counts):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    records = []
    for (suffix, head), count in zip(
        (("", "head_a"), ("O", "head_b"), ("N", "head_c")), head_counts
    ):
        for n in range(9, 9 + count):
            mol = Chem.MolFromSmiles("C" * n + suffix)
            canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
            records.append(
                {
                    "structure_id": "pl_" + hashlib.sha256(canonical.encode()).hexdigest(),
                    "morgan_fp_2048": DataStructs.BitVectToBinaryText(
                        generator.GetFingerprint(mol)
                    ),
                    "polar_id": head,
                    "chemistry_family": "phosphate",
                    "polar_plausibility_tier": "CORE",
                    "structure_origin": "pubchem_baseline",
                }
            )
    source, output = tmp_path / "structures.parquet", tmp_path / "audit"
    pq.write_table(pa.Table.from_pylist(records), source)
    cache = prepare_cache(source, tmp_path / "cache")
    benchmark = exact_benchmark(cache, tmp_path / "cache", 20, min(40, len(records)))
    full_readiness(cache, benchmark, tmp_path / "cache")
    result = full_head_audit(source, tmp_path / "cache", output, benchmark_targets=20, workers=1)
    assert result["status"] == "complete"
    expected_same = sum(n * min(n - 1, 10) for n in head_counts) / (len(records) * 10)
    assert (
        result["baseline_identity_only"]["same_head"]["same_polar_id_pair_fraction"]
        == expected_same
    )
    assert result["exclude_all_same_polar_id"]["same_head"]["same_polar_id_pair_fraction"] == 0.0
    expected_ten = sum(n for n in head_counts if len(records) - n >= 10) / len(records)
    assert result["exclude_all_same_polar_id"]["overall"]["ten_fraction"] == expected_ten
    if min(head_counts) < 10:
        assert result["exact_low_count_confirmations"]
        assert result["exact_low_count_confirmations"][0]["exact_count"] == 3
    assert result["baseline_metric_reproduction"]["template_count"]["equal_rows"] == len(records)
    assert (
        full_head_audit(source, tmp_path / "cache", output, benchmark_targets=20)[
            "baseline_identity_only"
        ]
        == result["baseline_identity_only"]
    )
