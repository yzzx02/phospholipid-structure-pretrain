"""Add the short-acyl/multiple-anchor risk check to the existing complete audit."""

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from structure_pretrain.glycerol_core_collection.collection import dump_json, file_hash
from structure_pretrain.headgroup_target_expansion.chemistry import (
    extra_glycerol_lipid_anchor,
    fatty_N_acyl_PE_PS_head,
)


def main():
    root = Path("outputs/headgroup_training_targets_v02_20261026")
    original = Path("data/structure_library_v0.3/polar_library.parquet")
    path = root / "background_195306_audit_P_first.parquet"
    summary = json.loads((root / "background_audit_summary.json").read_text(encoding="utf-8"))
    assert file_hash(original) == summary["original_SHA256"]
    source = pq.ParquetFile(original)
    previous = pq.ParquetFile(path)
    writer = None
    counts = Counter()
    passed = 0
    processed = 0
    for a, b in zip(
        source.iter_batches(batch_size=1024, columns=["polar_id", "polar_fragment_smiles"]),
        previous.iter_batches(batch_size=1024),
        strict=True,
    ):
        rows = []
        for src, row in zip(a.to_pylist(), b.to_pylist(), strict=True):
            assert src["polar_id"] == row["polar_id"]
            flags = set(filter(None, row["review_flags"].split(";")))
            mol = Chem.MolFromSmiles(src["polar_fragment_smiles"])
            if extra_glycerol_lipid_anchor(mol):
                flags.add("extra_glycerol_lipid_anchor_including_short_acyl")
            if fatty_N_acyl_PE_PS_head(mol):
                flags.add("extra_fatty_N_acyl_PE_PS_head")
            row["review_flags"] = ";".join(sorted(flags))
            row["future_500_600_policy_pass"] = not flags and row["formal_structure_pass"]
            passed += row["future_500_600_policy_pass"]
            counts.update(flags)
            rows.append(row)
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(
                path.with_suffix(".parquet.part"), table.schema, compression="zstd"
            )
        writer.write_table(table)
        processed += len(rows)
        if processed % 16384 == 0:
            print(json.dumps({"extra_anchor_audit_processed": processed}), flush=True)
    writer.close()
    previous.close()
    source.close()
    assert processed == 195306 and file_hash(original) == summary["original_SHA256"]
    path.with_suffix(".parquet.part").replace(path)
    summary["review_risk_counts_overlap"] = dict(counts)
    summary["check_counts"]["future_500_600_policy_pass"] = passed
    summary["short_chain_glycerol_anchor_check_complete"] = True
    dump_json(root / "background_audit_summary.json", summary)
    print(
        json.dumps(
            {
                "pass_new_scope": passed,
                "extra_glycerol_anchor": counts["extra_glycerol_lipid_anchor_including_short_acyl"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
