"""CPU-only integrity audit of all independent NEW v0.4 checkpoints (no weights published)."""

import json
from pathlib import Path

import torch

from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04 import SCHEMA
from structure_pretrain.polar_only_v04.prepare import dump


def main():
    torch.set_num_threads(4)
    root = Path("outputs/pretraining_v04")
    records = []
    assert not torch.cuda.is_initialized()
    for path in sorted((root / "runs").glob("*/checkpoint/step_*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=False)
        contract = state["contract"]
        assert contract["schema"] == SCHEMA and contract["version"] == "v0.4"
        assert contract["outputs"] == ["polar_SMILES_only"]
        assert contract["known_chains_required"]
        assert contract["train_targets"] <= 50000
        step = state["step"]
        assert path.stem == f"step_{step:06d}"
        assert len(state["history"]) == step and state["history"][-1]["step"] == step
        train = contract["config"]["train"]
        assert state["consumed"] == step * train["batch_size"] * train["gradient_accumulation"]
        assert all(int(v["step"]) == step for v in state["optimizer"]["state"].values())
        assert not any("chain_head" in name for name in state["model"])
        assert all(torch.isfinite(value).all() for value in state["model"].values())
        assert state["rng_cpu"].dtype == torch.uint8 and state["rng_gpu"]
        records.append(
            {
                "file": str(path),
                "step": step,
                "kind": contract["kind"],
                "encoder": contract["config"]["model"]["encoder"],
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
                "consumed_samples": state["consumed"],
                "finite_weights_optimizer_step_and_RNG_verified": True,
            }
        )
        print(json.dumps({"checkpoint": str(path), "status": "passed"}), flush=True)
        del state
    assert records and not torch.cuda.is_initialized()
    dump(
        root / "checkpoints_audit.json",
        {
            "status": "passed",
            "device": "cpu",
            "checkpoints": records,
            "count": len(records),
            "GPU_initialized": False,
        },
    )


if __name__ == "__main__":
    main()
