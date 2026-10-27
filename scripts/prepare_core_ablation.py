"""Freeze identical splits/labels and TRAIN-only formula statistics; no new selection."""

import os

import numpy as np
import pyarrow.parquet as pq

from structure_pretrain.core_ablation.common import (
    CONFIG,
    ROOT,
    TEMPLATE_ROOT,
    configuration,
    digest,
    read,
    verify_data,
    verify_historical,
    write,
)


def main():
    c = configuration()
    verify_historical()
    root, source = ROOT.resolve(), TEMPLATE_ROOT / "data"
    folder = root / "data"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists():
        verify_data(root)
        print("Frozen control data already verified", flush=True)
        return
    for name in ("tokens.npy", "token_lengths.npy", "splits.npy", "formulas.npy"):
        target = folder / name
        if not target.exists():
            os.link((source / name).resolve(), target)
        assert digest(target) == digest(source / name)
    splits = np.load(folder / "splits.npy")
    formulas = np.load(folder / "formulas.npy").astype(np.float64)
    train = formulas[splits == 0]
    mean, std = train.mean(0), train.std(0)
    std[std < 1e-6] = 1
    statistics = {
        "template_mean": mean.tolist(),
        "template_std": std.tolist(),
        "delta_mean": [0.0] * 10,
        "delta_std": [1.0] * 10,
        "fit_split": "TRAIN_target_formulas_only",
        "formula_count": len(train),
        "reference_features_used": False,
    }
    write(folder / "formula_statistics.json", statistics)
    test = np.flatnonzero(splits == 2)
    np.save(folder / "test_all.npy", test)
    p = formulas[test, 4] > 0
    assert len(test) == c["test_targets"] == 9499
    assert int(p.sum()) == c["P_test_targets"] and int((~p).sum()) == c["nonP_test_targets"]
    gold = pq.read_table(source / "targets_private.parquet", columns=["target_id", "split"])
    assert len(set(gold["target_id"].to_pylist())) == len(splits)
    assert all(gold["split"][int(i)].as_py() == "test" for i in test)
    original = read(source / "manifest.json")
    manifest = {
        "status": "passed",
        "configuration_sha256": digest(CONFIG),
        "original_data_manifest_sha256": digest(source / "manifest.json"),
        "split_counts": original["split_counts"],
        "targets": len(splits),
        "steps_per_epoch": original["steps_per_epoch"],
        "max_optimizer_steps": original["max_optimizer_steps"],
        "test_targets": len(test),
        "P_test_targets": int(p.sum()),
        "nonP_test_targets": int((~p).sum()),
        "bank_targets_remain_training_labels": 8000,
        "formula_only_reference_arrays_loaded": False,
        "assets_sha256": {f.name: digest(f) for f in sorted(folder.iterdir()) if f.is_file()},
    }
    assert manifest["split_counts"] == {"train": 170872, "validation": 9493, "test": 9499}
    write(folder / "manifest.json", manifest)
    verify_data(root)
    print(
        {
            "status": "passed",
            "split_counts": manifest["split_counts"],
            "test_P_nonP": [int(p.sum()), int((~p).sum())],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
