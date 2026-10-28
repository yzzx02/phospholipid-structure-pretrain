"""Unified Beam32 elements+grammar evaluation on all9493 frozen VALIDATION targets."""

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
from structure_pretrain.core_lr_study.common import (
    BASE_ROOT,
    ROOT,
    configuration,
    contract,
    digest,
    read,
    verify_data,
    write,
)
from structure_pretrain.core_lr_study.data import StudyData, make_model
from structure_pretrain.core_templates.metrics import aggregate, score_target
from structure_pretrain.core_templates.runtime import initialize
from structure_pretrain.utils.attention import attention_backend


def summarize(rows):
    result = aggregate(rows)
    total = len(rows) * 32
    actual = sum(r["raw"]["candidate_count"] for r in rows)
    valid = sum(r["raw"]["valid_candidates"] for r in rows)
    formula = sum(r["raw"]["formula_correct_candidates"] for r in rows)
    result["budget"] = {
        "beam_slots": total,
        "actual_candidates": actual,
        "unfilled_slots": total - actual,
        "core_valid_fraction_of_Beam32_budget": valid / total,
        "full_H_formula_correct_fraction_of_Beam32_budget": formula / total,
        "full_H_formula_correct_fraction_of_emitted_candidates": formula / max(actual, 1),
        "targets_with_formula_correct_candidate": sum(
            r["raw"]["formula_correct_candidates"] > 0 for r in rows
        )
        / len(rows),
        "RDKit_parse_valid_fraction_of_Beam32_budget": sum(r["RDKit_parse_count"] for r in rows)
        / total,
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=("E1", "E2", "E3", "E4"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, out, checkpoint = ROOT.resolve(), args.output.resolve(), args.checkpoint.resolve()
    binding, g = contract(root), configuration()["generation"]
    if out.parent != root or out.exists() or (root / "GPU_STOP.json").exists():
        raise ValueError("Fresh independent evaluation required")
    if not os.environ.get("LR_SUPERVISOR_PID") or not torch.cuda.is_available():
        raise ValueError("Supervised native CUDA required")
    best = read(root / f"{args.experiment}_seed42/best.json")
    assert checkpoint == Path(best["path"]).resolve() and digest(checkpoint) == best["sha256"]
    history = read(root / f"{args.experiment}_seed42/validation_history.json")
    assert best["best_validation_loss"] == min(r["validation_token_CE"] for r in history)
    assert len(history) >= 50
    out.mkdir()

    def event(**row):
        row["unix_time"] = time.time()
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    event(stage="verifying_immutable_data")
    verify_data(root)
    initialize(42)
    data, model = StudyData(root, "cuda"), make_model(root).cuda().eval()
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    expected = (
        read(BASE_ROOT / "queue_started.json")["contract"]
        if best.get("preserved_from_original_E3")
        else binding
    )
    assert state["contract"] == expected
    if not best.get("preserved_from_original_E3"):
        assert state["progress"]["experiment"] == args.experiment
    model.load_state_dict(state["model"])
    del state
    indices = np.load(root / "data/validation_indices.npy")
    targets = pq.read_table(root / "data/validation_targets_private.parquet").to_pylist()
    assert len(targets) == len(indices) == 9493
    assert all(t["split"] == "validation" for t in targets)
    assert np.all(data.splits[indices] == 1)
    baseline = np.load(BASE_ROOT / "data/similarities.npy")[:, 0]
    began, rows, classes, last_event = time.monotonic(), [], [], 0.0
    torch.cuda.reset_peak_memory_stats()
    with gzip.open(out / "raw_candidates_private.jsonl.gz", "wt", encoding="utf-8") as stream:
        with attention_backend("math"), torch.inference_mode():
            for start in range(0, len(indices), g["batch_size"]):
                selected = indices[start : start + g["batch_size"]]
                inputs = data.batch(selected)
                del inputs["target_polar_token_ids"]
                encoded = model.encode_templates(inputs)

                def heartbeat(position):
                    nonlocal last_event
                    if time.monotonic() - last_event > 5:
                        event(
                            stage="validation_decoding",
                            completed=start,
                            total=len(indices),
                            token=position,
                        )
                        last_event = time.monotonic()

                output = generate(
                    model.polar_decoder,
                    encoded["template_memory"],
                    encoded["memory_padding_mask"],
                    data.tokenizer,
                    inputs["target_core_formula"].cpu().numpy(),
                    mode="elements_grammar",
                    beam_size=32,
                    max_length=96,
                    progress=heartbeat,
                )
                for offset, (index, hypotheses) in enumerate(zip(selected, output)):
                    target = targets[start + offset]
                    candidates = [
                        {
                            "smiles": data.tokenizer.decode(r["ids"]),
                            "log_probability": r["log_probability"],
                            "eos": r["eos"],
                            "length_exhausted": r["length_exhausted"],
                        }
                        for r in hypotheses
                    ]
                    outcome = score_target(target, candidates, float(baseline[index]))
                    parsed_count = 0
                    for candidate in outcome["candidates"]:
                        try:
                            with rdBase.BlockLogs():
                                mol = Chem.MolFromSmiles(candidate["smiles"])
                            parsed = mol is not None and mol.GetNumAtoms() > 0
                        except Exception:
                            parsed = False
                        candidate["RDKit_parse_valid"] = parsed
                        parsed_count += parsed
                    outcome.update(
                        RDKit_parse_count=parsed_count,
                        contains_P=target["contains_P"],
                        row_index=int(index),
                    )
                    stream.write(json.dumps(outcome, ensure_ascii=False) + "\n")
                    rows.append({k: v for k, v in outcome.items() if k != "candidates"})
                    classes.append(target["contains_P"])
                stream.flush()
                event(
                    stage="validation_batch_completed",
                    completed=len(rows),
                    total=len(indices),
                    elapsed_seconds=time.monotonic() - began,
                )
    assert len(rows) == len({r["target_id"] for r in rows}) == 9493
    report = {
        **summarize(rows),
        "status": "completed",
        "experiment": args.experiment,
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_epoch": best["epoch"],
        "beam_size": 32,
        "decode_batch_size": g["batch_size"],
        "max_length": 96,
        "cohort": "all_9493_frozen_validation_targets",
        "test_used_for_selection": False,
        "elapsed_seconds": time.monotonic() - began,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "original_probabilities_not_renormalized": True,
        "online_heavy_atom_constraints": True,
        "online_SMILES_grammar": True,
        "full_H_formula_verified_finally": True,
        "template_retrieval_is_target_structure_oracle": True,
        "raw_means_online_constrained_before_posterior_filter": True,
        "unfilled_slots_count_as_failures_in_budget_rates": True,
    }
    for label, condition in (("P", True), ("nonP", False)):
        report[label] = summarize([r for r, p in zip(rows, classes) if p == condition])
    write(out / "evaluation_report.json", report)
    write(
        out / "evaluation_contract.json",
        {
            "contract": binding,
            "generation": g,
            "experiment": args.experiment,
            "checkpoint_sha256": digest(checkpoint),
            "raw_predictions_sha256": digest(out / "raw_candidates_private.jsonl.gz"),
        },
    )
    event(
        stage="validation_evaluation_completed",
        experiment=args.experiment,
        targets=len(rows),
        elapsed_seconds=report["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
