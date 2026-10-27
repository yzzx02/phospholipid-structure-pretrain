"""Full 9,499-target Beam20 evaluation; raw, elements and grammar remain distinct."""

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from rdkit import Chem, rdBase

from structure_pretrain.core_ablation.beam import generate
from structure_pretrain.core_ablation.common import (
    ROOT,
    TEMPLATE_ROOT,
    configuration,
    contract,
    digest,
    read,
    verify_data,
    write,
)
from structure_pretrain.core_ablation.data import FormulaData
from structure_pretrain.core_ablation.data import make_model as make_formula_model
from structure_pretrain.core_templates.data import CoreData
from structure_pretrain.core_templates.data import make_model as make_template_model
from structure_pretrain.core_templates.metrics import aggregate, score_target
from structure_pretrain.core_templates.runtime import initialize
from structure_pretrain.utils.attention import attention_backend


def summarize(rows, beam_size):
    result = aggregate(rows)
    total = len(rows) * beam_size
    actual = sum(r["raw"]["candidate_count"] for r in rows)
    valid = sum(r["raw"]["valid_candidates"] for r in rows)
    formula = sum(r["raw"]["formula_correct_candidates"] for r in rows)
    result["budget"] = {
        "beam_slots": total,
        "actual_candidates": actual,
        "unfilled_slots": total - actual,
        "core_valid_fraction_of_Beam20_budget": valid / total,
        "full_H_formula_correct_fraction_of_Beam20_budget": formula / total,
        "full_H_formula_correct_fraction_of_emitted_candidates": formula / max(actual, 1),
        "targets_with_formula_correct_candidate": sum(
            r["raw"]["formula_correct_candidates"] > 0 for r in rows
        )
        / len(rows),
        "RDKit_parse_valid_fraction_of_Beam20_budget": sum(r["RDKit_parse_count"] for r in rows)
        / total,
    }
    for k in (1, 5, 10):
        result["raw"][f"formula_correct_at_{k}"] = float(
            np.mean([r[f"formula_at_{k}"] for r in rows])
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("template", "formula_only"), required=True)
    parser.add_argument(
        "--decode-mode", choices=("raw", "elements", "elements_grammar"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out, checkpoint = ROOT.resolve(), args.output.resolve(), args.checkpoint.resolve()
    c, binding = configuration(), contract(root)
    if out.parent != root or out.exists() or (root / "GPU_STOP.json").exists():
        raise ValueError("Fresh dedicated supervised evaluation required")
    if not os.environ.get("ABLATION_SUPERVISOR_PID") or not torch.cuda.is_available():
        raise ValueError("Native supervised CUDA required")
    if args.model_kind == "template":
        assert checkpoint == Path(c["template_checkpoint"]).resolve()
        assert digest(checkpoint) == c["template_checkpoint_sha256"]
    else:
        pointer = read(checkpoint.parent.parent / "best.json")
        assert pointer["path"] == str(checkpoint) and digest(checkpoint) == pointer["sha256"]
    out.mkdir()

    def event(**row):
        row["unix_time"] = time.time()
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    event(stage="verifying_immutable_data")
    verify_data(root)
    initialize(c["seed"])
    if args.model_kind == "template":
        data, model = (
            CoreData(TEMPLATE_ROOT, "cuda"),
            make_template_model(TEMPLATE_ROOT).cuda().eval(),
        )
    else:
        data, model = FormulaData(root, "cuda"), make_formula_model(root).cuda().eval()
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    expected = (
        binding
        if args.model_kind == "formula_only"
        else read(TEMPLATE_ROOT / "queue_started.json")["contract"]
    )
    assert state["contract"] == expected
    model.load_state_dict(state["model"])
    del state
    indices = np.load(root / "data/test_all.npy")
    assert len(indices) == c["test_targets"] and np.all(data.splits[indices] == 2)
    targets = pq.read_table(
        TEMPLATE_ROOT / "data/targets_private.parquet",
        columns=[
            "target_id",
            "target_smiles",
            "core_formula",
            "contains_P",
        ],
    ).to_pylist()
    # This baseline enters metrics only after autoregressive decoding. It is never
    # passed to the Formula-only model or used to rank/filter generated hypotheses.
    baseline = np.load(TEMPLATE_ROOT / "data/similarities.npy")[:, 0]
    began, rows, classes, last_event = time.monotonic(), [], [], 0.0
    torch.cuda.reset_peak_memory_stats()
    with gzip.open(out / "raw_candidates_private.jsonl.gz", "wt", encoding="utf-8") as stream:
        with attention_backend("math"), torch.inference_mode():
            for start in range(0, len(indices), c["decode_batch_size"]):
                selected = indices[start : start + c["decode_batch_size"]]
                inputs = data.batch(selected)
                del inputs["target_polar_token_ids"]
                if args.model_kind == "formula_only":
                    assert set(inputs) == {"target_core_formula"}
                encoded = model.encode_templates(inputs)

                def heartbeat(position):
                    nonlocal last_event
                    now = time.monotonic()
                    if now - last_event > 5:
                        event(
                            stage="full_test_decoding",
                            completed=start,
                            total=len(indices),
                            mode=args.decode_mode,
                            token=position,
                        )
                        last_event = now

                output = generate(
                    model.polar_decoder,
                    encoded["template_memory"],
                    encoded["memory_padding_mask"],
                    data.tokenizer,
                    inputs["target_core_formula"].cpu().numpy(),
                    mode=args.decode_mode,
                    beam_size=c["beam_size"],
                    max_length=c["max_label_tokens"],
                    progress=heartbeat,
                )
                for index, hypotheses in zip(selected, output):
                    candidates = [
                        {
                            "smiles": data.tokenizer.decode(r["ids"]),
                            "log_probability": r["log_probability"],
                            "eos": r["eos"],
                            "length_exhausted": r["length_exhausted"],
                        }
                        for r in hypotheses
                    ]
                    outcome = score_target(targets[int(index)], candidates, float(baseline[index]))
                    rdkit_count = 0
                    for candidate in outcome["candidates"]:
                        try:
                            with rdBase.BlockLogs():
                                mol = Chem.MolFromSmiles(candidate["smiles"])
                            parsed = mol is not None and mol.GetNumAtoms() > 0
                        except Exception:
                            parsed = False
                        candidate["RDKit_parse_valid"] = parsed
                        rdkit_count += parsed
                    outcome["RDKit_parse_count"] = rdkit_count
                    for k in (1, 5, 10):
                        outcome[f"formula_at_{k}"] = int(
                            any(r["formula_match"] for r in outcome["candidates"][:k])
                        )
                    outcome["contains_P"] = targets[int(index)]["contains_P"]
                    outcome["row_index"] = int(index)
                    stream.write(json.dumps(outcome, ensure_ascii=False) + "\n")
                    rows.append({k: v for k, v in outcome.items() if k != "candidates"})
                    classes.append(outcome["contains_P"])
                stream.flush()
                event(
                    stage="full_test_batch_completed",
                    completed=len(rows),
                    total=len(indices),
                    elapsed_seconds=time.monotonic() - began,
                )
    assert len(rows) == len(indices) == len({r["target_id"] for r in rows}) == 9499
    assert sum(classes) == 8225 and len(classes) - sum(classes) == 1274
    report = {
        **summarize(rows, c["beam_size"]),
        "status": "completed",
        "model_kind": args.model_kind,
        "decode_mode": args.decode_mode,
        "checkpoint_sha256": digest(checkpoint),
        "beam_size": c["beam_size"],
        "decode_batch_size": c["decode_batch_size"],
        "max_length": c["max_label_tokens"],
        "test_cohort": "all_9499_frozen_test_targets",
        "test_used_for_selection": False,
        "elapsed_seconds": time.monotonic() - began,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "original_probabilities_not_renormalized": True,
        "online_heavy_atom_constraints": args.decode_mode != "raw",
        "online_SMILES_grammar": args.decode_mode == "elements_grammar",
        "full_H_formula_verified_finally": True,
        "formula_only_no_reference_inputs": args.model_kind == "formula_only",
        "gain_baseline": (
            "same frozen best nonself structural template; evaluation only for Formula-only"
        ),
        "template_retrieval_is_target_structure_oracle": args.model_kind == "template",
        "core_valid_definition": (
            "EOS, RDKit, correct rooted glycerol and port roles, scope, "
            "neutral reference-chain assembly"
        ),
        "unfilled_slots_count_as_failures_in_budget_rates": True,
    }
    for label, condition in (("P", True), ("nonP", False)):
        subset = [r for r, p in zip(rows, classes) if p == condition]
        report[label] = summarize(subset, c["beam_size"])
    write(out / "evaluation_report.json", report)
    write(
        out / "evaluation_contract.json",
        {
            "contract": binding,
            "configuration": c,
            "model_kind": args.model_kind,
            "decode_mode": args.decode_mode,
            "checkpoint_sha256": digest(checkpoint),
            "raw_predictions_sha256": digest(out / "raw_candidates_private.jsonl.gz"),
        },
    )
    event(
        stage="full_test_evaluation_completed",
        targets=len(rows),
        mode=args.decode_mode,
        model_kind=args.model_kind,
        elapsed_seconds=report["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
