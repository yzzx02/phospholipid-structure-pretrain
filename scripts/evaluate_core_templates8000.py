"""Stream an independently fixed whole-core cohort, without dropping RDKit failures."""

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from structure_pretrain.core_templates.common import (
    ROOT,
    configuration,
    contract,
    digest,
    read,
    verify_data,
    write,
)
from structure_pretrain.core_templates.data import CoreData, make_model
from structure_pretrain.core_templates.metrics import aggregate, score_target
from structure_pretrain.core_templates.runtime import initialize
from structure_pretrain.utils.attention import attention_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cohort", choices=("validation512", "bank_targets512", "test512"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out, checkpoint = ROOT.resolve(), args.output.resolve(), args.checkpoint.resolve()
    if out.parent != root or out.exists() or (root / "GPU_STOP.json").exists():
        raise ValueError("Fresh supervised whole-core evaluation output required")
    if not os.environ.get("CORE_SUPERVISOR_PID") or not torch.cuda.is_available():
        raise ValueError("Native CUDA supervision required")
    binding = contract(root)
    pointer = read(checkpoint.parent.parent / "best.json")
    if str(checkpoint) != pointer["path"] or digest(checkpoint) != pointer["sha256"]:
        raise ValueError("Evaluate only the frozen validation-CE selected best checkpoint")
    out.mkdir()

    def event(**row):
        row["unix_time"] = time.time()
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    event(stage="verifying_immutable_data")
    verify_data(root)
    initialize()
    data = CoreData(root, "cuda")
    model = make_model(root).cuda().eval()
    saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if saved["contract"] != binding:
        raise ValueError("Evaluation contract differs from the selected checkpoint")
    model.load_state_dict(saved["model"])
    del saved
    indices = np.load(root / "data" / (args.cohort + ".npy"))
    expected_split = (
        0 if args.cohort == "bank_targets512" else (1 if args.cohort == "validation512" else 2)
    )
    assert np.all(data.splits[indices] == expected_split)
    targets = pq.read_table(
        root / "data/targets_private.parquet",
        columns=[
            "target_id",
            "target_smiles",
            "core_formula",
            "contains_P",
        ],
    ).to_pylist()
    best_sim = np.load(root / "data/similarities.npy")[:, 0]
    began, outcomes, last_event = time.monotonic(), [], 0.0
    beam = configuration()["post_training_evaluation"]["beam_size"]
    with gzip.open(out / "raw_candidates_private.jsonl.gz", "wt", encoding="utf-8") as stream:
        with attention_backend("math"), torch.inference_mode():
            for number, index in enumerate(indices):

                def heartbeat(step):
                    nonlocal last_event
                    now = time.monotonic()
                    if now - last_event > 5:
                        event(
                            stage="decoding_progress",
                            completed=number,
                            targets=len(indices),
                            token=step,
                        )
                        last_event = now

                encoded = model.encode_templates(data.batch([index]))
                generated = model.polar_decoder.generate(
                    encoded["template_memory"],
                    encoded["memory_padding_mask"],
                    beam_size=beam,
                    max_length=configuration()["max_label_tokens"],
                    progress=heartbeat,
                )[0]
                candidates = [
                    {
                        "smiles": data.tokenizer.decode(r["ids"]),
                        "log_probability": r["log_probability"],
                        "eos": r["eos"],
                        "length_exhausted": r["length_exhausted"],
                    }
                    for r in generated
                ]
                outcome = score_target(targets[int(index)], candidates, float(best_sim[index]))
                stream.write(json.dumps(outcome, ensure_ascii=False) + "\n")
                stream.flush()
                outcomes.append({k: v for k, v in outcome.items() if k != "candidates"})
                event(
                    stage="target_completed",
                    completed=number + 1,
                    targets=len(indices),
                    contains_P=targets[int(index)]["contains_P"],
                    raw_exact10=outcome["raw"]["exact_at_10"],
                    filtered_candidates=outcome["posterior_filtered"][
                        "distinct_legal_formula_candidates"
                    ],
                )
    report = {
        **aggregate(outcomes),
        "status": "completed",
        "cohort": args.cohort,
        "cohort_note": "TRAIN bank-target leave-one-out reconstruction"
        if expected_split == 0
        else "held-out grouped targets",
        "checkpoint_sha256": digest(checkpoint),
        "beam_size": beam,
        "elapsed_seconds": time.monotonic() - began,
        "formula_basis": configuration()["formula_scope"],
        "self_equivalent_templates_excluded": True,
        "test_used_for_model_selection": False,
        "retrieval": (
            "target-structure oracle for structural pretraining; not an end-to-end spectra test"
        ),
    }
    for label, p in (("P", True), ("nonP", False)):
        subset = [r for r, i in zip(outcomes, indices) if targets[int(i)]["contains_P"] == p]
        report[label] = aggregate(subset) if subset else {"targets": 0}
    write(out / "evaluation_report.json", report)
    event(
        stage="evaluation_completed",
        targets=len(indices),
        elapsed_seconds=report["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
