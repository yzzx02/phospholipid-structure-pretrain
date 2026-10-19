"""Independent canonical/marker/P-boundary/composition audit of every cached variant."""

from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from rdkit import Chem

from structure_pretrain.data.prepare_v03 import atom_counts, read_json
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.features import (
    fragment_features,
    fragment_molecule,
    packed_fingerprint,
)
from structure_pretrain.polar_only_v04.prepare import DEFAULT_ROOT, V03, dump, emit


def main():
    root = DEFAULT_ROOT
    manifest = read_json(root / "features_manifest.json")
    counts = np.load(root / "head_polar_formulas.npy", mmap_mode="r")
    scaffold = np.load(root / "scaffold_counts.npy")
    fp = np.load(root / "chainless_fingerprints.npy", mmap_mode="r")
    family = pq.read_table(V03 / "polar_groups.parquet")["chemistry_family"].to_pylist()
    sample = set(np.random.default_rng(42).choice(len(fp), 512, replace=False).tolist())
    families, checked = Counter(), 0
    for shard in manifest["shards"]:
        path = root / shard["file"]
        assert file_hash(path) == shard["sha256"]
        with pa.memory_map(str(path), "r") as handle:
            records = ipc.open_file(handle).read_all().to_pylist()
        for record in records:
            v, g, occupancy = record["variant"], record["head"], record["occupancy"]
            mol = fragment_molecule(record["fragment_smiles"])
            assert (
                Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                == record["fragment_smiles"]
            )
            expected = scaffold + counts[g]
            expected[1] += occupancy == 1
            np.testing.assert_array_equal(atom_counts(mol), expected)
            if v in sample:
                actual = fragment_features(mol)
                assert all(actual[key] == record[key] for key in actual)
                np.testing.assert_array_equal(packed_fingerprint(mol), fp[v])
            families[f"{family[g]}/chains{occupancy}"] += 1
            checked += 1
        emit({"stage": "all_variant_canonical_marker_formula_audit", "variants": checked})
    assert checked == len(fp)
    dump(
        root / "fragment_integrity_audit.json",
        {
            "status": "passed",
            "variants": checked,
            "all_shard_hashes_verified": True,
            "canonical_roundtrip_failures": 0,
            "dummy_or_glycerol_O_P_failures": 0,
            "formula_failures": 0,
            "independent_graph_and_FP_recomputations": len(sample),
            "family_occupancy_counts": dict(families),
            "features_manifest_sha256": file_hash(root / "features_manifest.json"),
        },
    )


if __name__ == "__main__":
    main()
