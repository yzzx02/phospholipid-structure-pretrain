"""CPU-only immutable initialization, original E3 epoch48 and validation selection audit."""

import json

import numpy as np
import pyarrow.parquet as pq
import torch

from structure_pretrain.core_lr_study.common import (
    BASE_ROOT,
    CONFIG,
    ROOT,
    configuration,
    digest,
    original_run,
    parameter_digest,
    read,
    verify_data,
    verify_historical,
    write,
)
from structure_pretrain.core_templates.data import make_model
from structure_pretrain.core_templates.runtime import initialize, rng_state


def main():
    c = configuration()
    verify_historical()
    folder = ROOT / "data"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists():
        verify_data()
        print("Existing frozen LR study input verified", flush=True)
        return
    initialize(c["seed"])
    model = make_model(BASE_ROOT)
    init = {
        "model": model.state_dict(),
        "rng": rng_state("cpu"),
        "seed": 42,
        "parameter_sha256": parameter_digest(model),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initialization_recipe": "unchanged original initialize(42) and make_model; no data RNG "
        "consumed",
    }
    target = folder / "shared_initial_state.pt"
    if target.exists():
        existing = torch.load(target, map_location="cpu", weights_only=False)
        assert existing["parameter_sha256"] == init["parameter_sha256"]
    else:
        temporary = target.with_suffix(".partial")
        torch.save(init, temporary)
        check = torch.load(temporary, map_location="cpu", weights_only=False)
        assert check["parameter_sha256"] == init["parameter_sha256"]
        temporary.replace(target)
    original = original_run()
    history = read(original / "validation_history.json")
    assert [r["epoch"] for r in history] == list(range(1, 49))
    assert all(r["validation_targets"] == 9493 for r in history)
    best, last = read(original / "best.json"), read(original / "last.json")
    assert last["epoch"] == 48 and last["step"] == 256320
    assert digest(last["path"]) == c["original_E3_epoch48_sha256"]
    assert digest(best["path"]) == c["original_E3_best_sha256"]
    checkpoint = torch.load(last["path"], map_location="cpu", weights_only=False)
    assert checkpoint["contract"] == read(BASE_ROOT / "queue_started.json")["contract"]
    assert checkpoint["progress"]["epoch"] == 48 and checkpoint["progress"]["batch_cursor"] == 0
    assert checkpoint["progress"]["step"] == checkpoint["scheduler"]["updates_completed"] == 256320
    assert checkpoint["scheduler"]["specification"]["total_steps"] == 427200
    assert checkpoint["progress"]["validation_history"] == history
    split = np.load(BASE_ROOT / "data/splits.npy")
    train, validation = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    np.save(folder / "train_indices.npy", train)
    np.save(folder / "validation_indices.npy", validation)
    # Filter in Arrow before exposing rows to the evaluator. No test target or
    # test generation metric is read or used in this new schedule study.
    table = pq.read_table(
        BASE_ROOT / "data/targets_private.parquet",
        filters=[("split", "=", "validation")],
        columns=["target_id", "target_smiles", "core_formula", "contains_P", "split"],
    )
    assert len(table) == len(validation) == 9493
    pq.write_table(table, folder / "validation_targets_private.parquet", compression="zstd")
    write(folder / "baseline_E3_history.json", history)
    write(folder / "baseline_E3_best.json", best)
    write(folder / "baseline_E3_last.json", last)
    write(
        folder / "initialization_audit.json",
        {k: v for k, v in init.items() if k not in ("model", "rng")},
    )
    assets = {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()}
    manifest = {
        "status": "passed",
        "configuration_sha256": digest(CONFIG),
        "base_data_manifest_sha256": digest(BASE_ROOT / "data/manifest.json"),
        "assets_sha256": assets,
        "train_targets": len(train),
        "validation_targets": len(validation),
        "steps_per_epoch": 5340,
        "schedule_total_steps": 427200,
        "minimum_epochs": 50,
        "max_epochs": 80,
        "initial_parameter_sha256": init["parameter_sha256"],
        "parameter_count": init["parameter_count"],
        "original_E3_epochs": 48,
        "original_E3_last_sha256": last["sha256"],
        "test_split_used": False,
        "original_E3_modified": False,
        "baseline_original_snapshot_exists": False,
        "baseline_initialization_evidence": (
            "frozen recorded seed and identical construction recipe; old "
            "initial tensor snapshot was not stored"
        ),
    }
    write(folder / "manifest.json", manifest)
    verify_data()
    print(
        json.dumps(
            {
                "status": "passed",
                "train": len(train),
                "validation": len(validation),
                "initial_sha256": init["parameter_sha256"],
                "parameters": init["parameter_count"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
