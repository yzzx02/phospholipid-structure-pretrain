"""CPU-only weight/AdamW/data-position inspection; never executes a training step."""

import argparse
import json

import numpy as np
import torch
from evaluate_polar_similarity_v04 import CHECKPOINTS, ROOT

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.fp50k_data import FEATURES, get_contract
from structure_pretrain.polar_only_v04.fp50k_data import ROOT as TRAIN_ROOT
from structure_pretrain.polar_only_v04.prepare import SAMPLES, V03, row_hash
from structure_pretrain.polar_only_v04.runtime import load_config, make_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="after_cleanup")
    args = p.parse_args()
    torch.set_num_threads(4)
    tokenizer = SmilesTokenizer(read_json(V03 / "polar_tokenizer.json")["tokens"])
    results = {}
    for key, path in CHECKPOINTS.items():
        config = load_config(
            "configs/polar_only_v04_fp.yaml"
            if key.startswith("20k")
            else "configs/polar_fp_base_50k.yaml"
        )
        state = torch.load(path, map_location="cpu", weights_only=False)
        if key.startswith("50k"):
            assert state["contract"] == get_contract(TRAIN_ROOT, True)[2]
        else:
            assert state["contract"]["target_rows_sha256"] == row_hash(
                np.load(SAMPLES / "train_pool.npy")
            )
            assert state["contract"]["train_targets"] == 20000
        model = make_model(config, tokenizer, FEATURES)
        model.load_state_dict(state["model"], strict=True)
        count = sum(x.numel() for x in model.parameters())
        assert count == 16468671 and all(torch.isfinite(x).all() for x in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        optimizer.load_state_dict(state["optimizer"])
        steps = sorted({int(s["step"]) for s in optimizer.state.values()})
        assert steps == [state["step"]] and state["consumed"] == state["step"] * 16
        results[key] = {
            "checkpoint_sha256": file_hash(path),
            "parameters": count,
            "strict_CPU_model_load": True,
            "CPU_AdamW_state_load": True,
            "optimizer_step_values": steps,
            "consumed": state["consumed"],
            "CPU_RNG_bytes": state["rng_cpu"].numel(),
            "GPU_RNG_state_bytes": [x.numel() for x in state["rng_gpu"]],
            "all_required_RNG_fields_present": all(
                k in state for k in ("rng_cpu", "rng_gpu", "rng_numpy", "rng_python")
            ),
            "training_steps_executed": 0,
        }
        del state, model, optimizer
    output = ROOT / f"checkpoint_load_{args.label}.json"
    with output.open("x", encoding="utf-8") as f:
        json.dump({"passed": True, "models": results, "training_started": False}, f, indent=2)
    print(json.dumps({"CPU_models_loaded": 3, "optimizer_steps_executed": 0}))


if __name__ == "__main__":
    main()
