"""Read-only hash gate for copied v0.3 data and derived training assets."""

import argparse
import json
from pathlib import Path

from structure_pretrain.library_builder.pipeline import file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(prepared, library, frozen, shards=True):
    prepared, library, frozen = map(Path, (prepared, library, frozen))
    label = read(prepared / "labels_manifest.json")
    inputs = read(prepared / "input_manifest.json")
    if label["status"] != "complete" or label["version"] != "v0.3":
        raise ValueError("v0.3 completed labels required")
    for name, digest in inputs["inputs_sha256"].items():
        if file_hash(library / name) != digest:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    if file_hash(frozen / "morgan_groups.usearch") != inputs["frozen_index_sha256"]:
        raise ValueError("Frozen v0.3 index changed")
    if (
        read(frozen / "fingerprint_manifest.json")["structures_sha256"]
        != label["structures_sha256"]
    ):
        raise ValueError("Frozen fingerprint source mismatch")
    checked = 0
    frozen_files = prepared / "frozen_index_files.json"
    if frozen_files.exists():
        for name, digest in read(frozen_files)["files_sha256"].items():
            if file_hash(frozen / name) != digest:
                raise ValueError(f"Frozen cache hash mismatch: {name}")
            checked += 1
    for name, digest in label["derived_sha256"].items():
        if file_hash(prepared / name) != digest:
            raise ValueError(f"Derived label/split hash mismatch: {name}")
        checked += 1
    if shards:
        for item in label["shards"]:
            if file_hash(prepared / item["file"]) != item["sha256"]:
                raise ValueError(f"Arrow shard hash mismatch: {item['file']}")
            checked += 1
    coverage = read(prepared / "retrieval_coverage_by_split.json")
    bank = read(prepared / "train_only_bank_manifest.json")
    if coverage["status"] != "complete" or bank["held_out_reference_rows"] != 0:
        raise ValueError("Completed train-only retrieval required")
    if file_hash(prepared / "train_bank/train_groups.usearch") != bank["index_sha256"]:
        raise ValueError("Train bank index hash mismatch")
    if file_hash(prepared / "labels_manifest.json") != bank["labels_manifest_sha256"]:
        raise ValueError("Train bank belongs to different labels")
    for name, digest in bank["derived_sha256"].items():
        if file_hash(prepared / "train_bank" / name) != digest:
            raise ValueError(f"Train postings hash mismatch: {name}")
        checked += 1
    for mode in ("normal", "polar_excluded"):
        folder = prepared / f"retrieval_{mode}"
        manifest = read(folder / "manifest.json")
        for filename, key in (
            ("template_rows.npy", "rows_sha256"),
            ("scores.npy", "scores_sha256"),
        ):
            if file_hash(folder / filename) != manifest[key]:
                raise ValueError(f"Retrieval hash mismatch: {mode}/{filename}")
            checked += 1
        if manifest["index_sha256"] != bank["index_sha256"]:
            raise ValueError("Retrieval bank mismatch")
    stats = read(prepared / "formula_stats_train_only.json")
    if (
        stats["structures_sha256"] != label["structures_sha256"]
        or stats["bank_index_sha256"] != bank["index_sha256"]
        or not stats["training_pairs_only"]
    ):
        raise ValueError("Formula statistics provenance mismatch")
    return {
        "status": "passed",
        "version": "v0.3",
        "checked_derived_files": checked,
        "shards_checked": shards,
        "structures_sha256": label["structures_sha256"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="data/private/pretraining_v03")
    p.add_argument("--library", default="data/structure_library_v0.3")
    p.add_argument("--frozen-index", default="data/private/morgan_index_v0.3")
    a = p.parse_args()
    print(json.dumps(verify(a.prepared, a.library, a.frozen_index), indent=2))
