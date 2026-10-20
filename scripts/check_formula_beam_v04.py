"""Run CPU gates before the separately supervised, inference-only GPU jobs."""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.polar_only_v04.formula_beam_eval import EXPECTED_COHORT, EXPERIMENT


def verified_gate(experiment=EXPERIMENT):
    log = (experiment / "all_tests_cpu.log").read_text()
    match = re.search(r"(?:^|\n)(\d+) passed in [\d.]+s", log)
    if not match or re.search(r"\d+ (?:failed|errors?)", log):
        raise ValueError("CPU pytest evidence did not pass")
    pip_log = experiment / "pip_check_cpu.log"
    if pip_log.read_text().strip() != "No broken requirements found.":
        raise ValueError("pip check did not pass")
    audits = [
        json.loads((experiment / f"cpu_{key}_audit/cpu_audit.json").read_text())
        for key in ("fp", "gine")
    ]
    if any(
        not audit["passed"]
        or audit["errors"]
        or audit["checked_targets"] != 512
        or audit["cohort_rows_sha256"] != EXPECTED_COHORT
        or audit["device"] != "cpu"
        or audit["inputs"]["step"] != 3000
        for audit in audits
    ):
        raise ValueError("Both frozen real-checkpoint CPU audits are mandatory")
    roundtrip = json.loads((experiment / "target_roundtrip_audit.json").read_text())
    if not roundtrip["passed"] or roundtrip["complete_formula_and_graph_roundtrip_passed"] != 512:
        raise ValueError("All target formula and graph round trips must pass")
    return {
        "passed": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "tests_passed": int(match[1]),
        "pip_check": "passed",
        "cpu_checkpoint_audits": audits,
        "target_roundtrip_audit": roundtrip,
        "training_started": False,
        "evidence_sha256": {
            path.name: file_hash(path) for path in (experiment / "all_tests_cpu.log", pip_log)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true", help="Read-only evidence check")
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT)
    parser.add_argument("--gpu-backend", choices=("rocm", "native_cuda"), default="rocm")
    args = parser.parse_args()
    experiment = args.experiment_root
    protected = (EXPERIMENT.resolve(), Path("outputs/pretraining_v04").resolve())
    expected = Path("/home/administrator/miniconda3/envs/torch-rocm721/bin/python")
    if args.gpu_backend == "rocm" and Path(sys.executable).resolve() != expected.resolve():
        raise ValueError(f"Use the verified interpreter: {expected}")
    if args.gpu_backend == "native_cuda" and (
        sys.platform != "win32"
        or not experiment.resolve().is_relative_to(Path("outputs").resolve())
        or any(
            experiment.resolve() == p
            or experiment.resolve().is_relative_to(p)
            or p.is_relative_to(experiment.resolve())
            for p in protected
        )
    ):
        raise ValueError("Native Windows checks require a separate outputs/ experiment root")
    if not args.verify_existing:
        if (experiment / "cpu_gate.json").exists():
            raise ValueError("Existing CPU gate is protected; use a fresh experiment workspace")
        experiment.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            HSA_ENABLE_DXG_DETECTION="0",
            CUDA_VISIBLE_DEVICES="-1",
            HIP_VISIBLE_DEVICES="-1",
            ROCR_VISIBLE_DEVICES="-1",
        )
        commands = [
            (["-m", "pytest", "-q"], "all_tests_cpu.log"),
            (["-m", "pip", "check"], "pip_check_cpu.log"),
        ]
        for key, encoder in (("fp", "fingerprint"), ("gine", "graph")):
            commands.append(
                (
                    [
                        "-u",
                        "-m",
                        "structure_pretrain.polar_only_v04.formula_beam_eval",
                        "--encoder",
                        encoder,
                        "--audit-only",
                        "--device",
                        "cpu",
                        "--output",
                        str(experiment / f"cpu_{key}_audit"),
                        "--experiment-root",
                        str(experiment),
                        "--gpu-backend",
                        args.gpu_backend,
                    ],
                    f"cpu_{key}_audit.log",
                )
            )
        commands.append(
            (
                ["scripts/audit_formula_targets_v04.py", "--experiment-root", str(experiment)],
                "target_roundtrip_audit.log",
            )
        )
        for command, name in commands:
            print(json.dumps({"stage": "CPU_gate", "log": name}), flush=True)
            with (experiment / name).open("x", encoding="utf-8") as stream:
                subprocess.run(
                    [sys.executable, *command],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    gate = verified_gate(experiment)
    gate.update(gpu_backend=args.gpu_backend, interpreter=str(Path(sys.executable).resolve()))
    if args.gpu_backend == "native_cuda":
        gate["inference_source_sha256"] = {
            name: file_hash(Path(name))
            for name in (
                "src/structure_pretrain/polar_only_v04/formula_beam_eval.py",
                "src/structure_pretrain/polar_only_v04/constrained_decode.py",
                "src/structure_pretrain/polar_only_v04/runtime.py",
                "src/structure_pretrain/models/polar_decoder.py",
                "src/structure_pretrain/training/quality_v03.py",
                "src/structure_pretrain/utils/resources.py",
                "src/structure_pretrain/utils/attention.py",
                "scripts/supervise_formula_beam_cuda_v04.py",
            )
        }
    if not args.verify_existing:
        with (experiment / "cpu_gate.json").open("x") as stream:
            stream.write(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"passed": gate["passed"], "tests_passed": gate["tests_passed"]}))


if __name__ == "__main__":
    main()
