"""CPU real-data resume equivalence gate and explicit real CUDA probe receipt."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from supervise_formula_beam_cuda_v04 import digest, read, write_new

from structure_pretrain.polar_only_v04.fp50k_data import ROOT, get_contract


def compare(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare(a, b)
    else:
        assert left == right


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, default=ROOT)
    p.add_argument("--gpu-result", action="store_true")
    args = p.parse_args()
    root = args.experiment_root
    _, _, contract = get_contract(root, verify_assets=True)
    if args.gpu_result:
        for name, start, end in (("gpu_probe_20", 0, 20), ("gpu_probe_resume_25", 20, 25)):
            metrics = read(root / name / "training_metrics.json")
            host = read(root / name / "host_monitor.json")
            assert metrics["contract"] == contract and metrics["device"] == "cuda"
            assert metrics["probe"] and metrics["step_start"] == start and metrics["step"] == end
            assert metrics["optimizer_steps_verified"] and metrics["checkpoint_reload_logits_close"]
            assert all(metrics["gradients_finite_nonzero"].values())
            assert all(v["changed_parameters"] > 0 for v in metrics["module_updates"].values())
            assert host["stop_reason"] is None and host["child_exit_code"] == 0
        write_new(
            root / "gpu_training_gate.json",
            {
                "passed": True,
                "contract": contract,
                "real_cuda_forward_backward_steps": 25,
                "real_resume_from_step": 20,
                "probe_metrics_sha256": {
                    name: digest(root / name / "training_metrics.json")
                    for name in ("gpu_probe_20", "gpu_probe_resume_25")
                },
            },
        )
        print("Real CUDA backward and optimizer/RNG/data resume gate passed")
        return
    if (root / "cpu_gate.json").exists():
        raise FileExistsError("CPU gate must be fresh")
    with (root / "cpu_gate.log").open("x", encoding="utf-8") as log:
        for command in (
            [sys.executable, "-m", "pip", "check"],
            [sys.executable, "-m", "pytest", "-q"],
        ):
            subprocess.run(command, stdout=log, stderr=log, check=True)
        for name, steps, resume in (
            ("cpu_continuous_4", 4, None),
            ("cpu_split_2", 2, None),
            ("cpu_resume_4", 4, root / "cpu_split_2/checkpoint/step_000002.pt"),
        ):
            command = [
                sys.executable,
                "train_polar_fp50k.py",
                "--device",
                "cpu",
                "--probe",
                "--experiment-root",
                str(root),
                "--output",
                str(root / name),
                "--steps",
                str(steps),
            ]
            if resume:
                command += ["--resume", str(resume)]
            subprocess.run(command, stdout=log, stderr=log, check=True)
    a = torch.load(
        root / "cpu_continuous_4/checkpoint/step_000004.pt", map_location="cpu", weights_only=False
    )
    b = torch.load(
        root / "cpu_resume_4/checkpoint/step_000004.pt", map_location="cpu", weights_only=False
    )
    for key in ("contract", "model", "optimizer", "rng_cpu", "rng_python", "step", "consumed"):
        compare(a[key], b[key])
    assert a["consumed"] == 64
    assert [r["polar_CE"] for r in a["history"]] == [r["polar_CE"] for r in b["history"]]
    gate = {
        "passed": True,
        "interpreter": str(Path(sys.executable).resolve()),
        "contract": contract,
        "pip_check": True,
        "pytest": True,
        "real_cpu_backward_steps": 8,
        "resume_equivalence": "4 continuous vs 2+2 resumed: exact weights, optimizer, "
        "CPU/Python RNG, consumption and loss sequence",
        "log_sha256": digest(root / "cpu_gate.log"),
    }
    write_new(root / "cpu_gate.json", gate)
    print(json.dumps({"passed": True, "resume_exact": True}))


if __name__ == "__main__":
    main()
