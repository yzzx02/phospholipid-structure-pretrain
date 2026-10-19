"""Read-only CPU audit of optimizer counters and immutable resume contracts."""

import json
from pathlib import Path

import torch

from structure_pretrain.library_builder.pipeline import file_hash


def main():
    torch.set_num_threads(4)
    root = Path("outputs/midterm_v03")
    reports = []
    for encoder in ("fingerprint", "graph"):
        original = Path(f"outputs/pretraining_v03/rocm_smoke/{encoder}_resume/checkpoint/last.pt")
        baseline = torch.load(original, map_location="cpu", weights_only=False)
        contract = baseline["contract"]
        assert baseline["step"] == 1001
        baseline_hash = file_hash(original)
        for report_path in (root / "evaluation").glob("*/quality_report.json"):
            report = json.loads(report_path.read_text())
            if report["encoder"] == encoder and report["step"] == 1001:
                assert baseline_hash == report["checkpoint_sha256"]
        for path in sorted((root / "training").glob("*/checkpoint/step_*.pt")):
            phase = json.loads((path.parents[1] / "training_metrics.json").read_text())
            if phase["contract"]["model_config"]["encoder"] != encoder:
                continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            assert state["contract"] == contract
            step = int(state["step"])
            assert step == int(path.stem.split("_")[-1])
            assert state["consumed"] == step * 16
            assert len(state["history"]) == step
            counters = [int(s["step"]) for s in state["optimizer"]["state"].values()]
            assert counters and min(counters) == max(counters) == step
            assert all(torch.isfinite(t).all() for t in state["model"].values())
            assert state["rng_torch"].dtype == torch.uint8
            assert all(t.dtype == torch.uint8 for t in state["rng_device"])
            groups = state["optimizer"]["param_groups"]
            assert all(g["lr"] == contract["train_config"]["learning_rate"] for g in groups)
            reports.append(
                {
                    "encoder": encoder,
                    "step": step,
                    "checkpoint_sha256": file_hash(path),
                    "baseline_1001_sha256": baseline_hash,
                    "contract_equal_to_original_1001": True,
                    "optimizer_parameter_states": len(counters),
                    "optimizer_step_min_max": [min(counters), max(counters)],
                    "consumed_targets": state["consumed"],
                    "effective_batch_size": 16,
                    "model_tensors_finite": True,
                    "saved_rng_byte_tensors_valid": True,
                }
            )
            del state
        assert file_hash(original) == baseline_hash
        del baseline
    result = {
        "status": "passed",
        "checkpoints": reports,
        "original_step1001_files_unchanged": True,
        "checkpoint_files_published": False,
    }
    (root / "checkpoint_integrity_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": "passed", "independent_checkpoints_audited": len(reports)}))


if __name__ == "__main__":
    main()
