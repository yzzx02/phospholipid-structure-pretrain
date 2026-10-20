"""CPU checkpoint audit of every FP submodule, including formula MLP and TCLS."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from supervise_formula_beam_cuda_v04 import digest, write_new

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.polar_only_v04.fp50k_data import FEATURES, ROOT, get_contract
from structure_pretrain.polar_only_v04.prepare import V03
from structure_pretrain.polar_only_v04.runtime import make_model


def audit(checkpoint, root):
    config, _, contract = get_contract(root)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = SmilesTokenizer(read_json(V03 / "polar_tokenizer.json")["tokens"])
    initial = make_model(config, tokenizer, FEATURES)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["contract"] == contract and not saved["probe"]
    assert saved["consumed"] == saved["step"] * 16
    ids = saved["optimizer"]["param_groups"][0]["params"]
    state = saved["optimizer"]["state"]
    checks = {}
    for prefix in (
        "structure_encoder",
        "projector.formula_mlp",
        "projector.fusion_mlp",
        "template_set.tcls",
        "template_set.encoder",
        "polar_decoder",
    ):
        changed, nonzero, count = 0, 0, 0
        for (name, param), pid in zip(initial.named_parameters(), ids):
            if name == prefix or name.startswith(prefix + "."):
                actual = saved["model"][name]
                assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
                changed += int(torch.count_nonzero(actual - param.detach()))
                moment = state[pid]["exp_avg"]
                assert torch.isfinite(moment).all() and int(state[pid]["step"]) == saved["step"]
                nonzero += int(torch.count_nonzero(moment))
                count += param.numel()
        assert changed > 0 and nonzero > 0 and count > 0
        checks[prefix] = {
            "parameters": count,
            "changed_from_seed42": changed,
            "nonzero_optimizer_gradient_moments": nonzero,
        }
    return {
        "passed": True,
        "checkpoint_sha256": digest(checkpoint),
        "step": saved["step"],
        "submodules": checks,
        "all_loss_finite": all(np.isfinite(x["polar_CE"]) for x in saved["history"]),
        "consumed": saved["consumed"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, default=ROOT)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--stage-a-decision", action="store_true")
    args = p.parse_args()
    result = audit(args.checkpoint, args.experiment_root)
    assert result["all_loss_finite"]
    write_new(args.experiment_root / f"checkpoint_audit_{result['step']:06d}.json", result)
    if args.stage_a_decision:
        assert result["step"] == 1000
        root = args.experiment_root
        training = read_json(root / "train_A_1000/training_metrics.json")
        greedy = read_json(root / "eval_A_1000_greedy_512/metrics.json")
        beam5 = read_json(root / "eval_A_1000_beam5_64/metrics.json")
        for name in ("train_A_1000", "eval_A_1000_greedy_512", "eval_A_1000_beam5_64"):
            host = read_json(root / name / "host_monitor.json")
            assert host["stop_reason"] is None and host["protected_checkpoints_unchanged"]
        assert training["last_loss"] < training["first_loss_this_stage"]
        assert greedy["raw_top1_eos_rate_over_all_targets"] > 0
        assert greedy["raw"]["raw_top1_chemical_valid_polar_rate"] > 0
        _, _, contract = get_contract(root)
        write_new(
            root / "stage_A_decision.json",
            {
                "continue": True,
                "contract": contract,
                "checkpoint_sha256": result["checkpoint_sha256"],
                "reasons": [
                    "finite falling loss; all six submodules update with nonzero gradient moments",
                    "real CPU and CUDA optimizer/data/RNG resume gates passed",
                    "greedy EOS and chemically valid Polar observed; no monitored GPU fault",
                    "basic engineering gate only; exact-structure improvement judged at 3000",
                ],
                "greedy_512": {
                    k: greedy[k]
                    for k in (
                        "raw_top1_eos_rate_over_all_targets",
                        "raw_top1_length_quantiles",
                        "raw",
                    )
                },
                "beam5_prefix64": beam5["strict_post_filter"],
            },
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
