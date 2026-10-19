"""Independent fixed-cohort evaluation; never mutates a training checkpoint or RNG state."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from structure_pretrain.config import ExperimentConfig
from structure_pretrain.data.batching import collate_structure_examples
from structure_pretrain.data.prepare_v03 import canonical_polar, read_json
from structure_pretrain.data.real_structure_dataset import RealStructureDataset
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel
from structure_pretrain.training.quality_v03 import aggregate, load_builder_rules, score_candidate
from structure_pretrain.utils.attention import attention_backend
from train_structure_real import move_batch, peak_ram_bytes


@torch.no_grad()
def evaluate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostic.pid").write_text(str(os.getpid()))

    def progress(record):
        with (output / "events.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    started = time.monotonic()
    progress({"stage": "loading_checkpoint"})
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Working ROCm GPU required")
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_step = int(saved["step"])
    contract = saved["contract"]
    assert contract.get("attention_backend") == args.attention_backend == "math"
    assert contract["run"]["precision"] == "fp32"
    selection_root = Path(args.manifest).parent
    manifest = read_json(args.manifest)
    cohorts = {
        name: np.load(selection_root / f"{name}.npy")
        for name in ["smoke_128", "diagnostic_512", "validation_pool"]
    }
    for name, rows in cohorts.items():
        assert (
            hashlib.sha256(rows.tobytes()).hexdigest() == manifest["cohorts"][name]["rows_sha256"]
        )
    assert manifest["cohorts"]["train_pool"]["rows_sha256"] == contract["train_targets_sha256"]
    assert (
        manifest["cohorts"]["validation_pool"]["rows_sha256"]
        == contract["validation_targets_sha256"]
    )
    for key, filename in [
        ("labels_manifest_sha256", "labels_manifest.json"),
        ("split_sha256", "row_splits.npy"),
    ]:
        assert manifest[key] == file_hash(Path(args.prepared) / filename)
    config = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(ExperimentConfig()),
            {
                "model": contract["model_config"],
                "train": contract["train_config"],
                "retrieval": contract["retrieval_config"],
            },
        )
    )
    all_rows = np.unique(np.concatenate(list(cohorts.values())))
    data = RealStructureDataset(
        args.prepared,
        args.frozen_index,
        "validation",
        config.model.encoder,
        "polar_excluded",
        all_rows,
        feature_cache_size=1024,
    )
    model = StructurePretrainModel(
        config.model,
        len(data.chain_vocab),
        data.tokenizer,
        read_json(Path(args.prepared) / "formula_stats_train_only.json"),
    )
    model.load_state_dict(saved["model"])
    model = model.to("cuda").eval()
    del saved
    teacher = {}
    for start in range(0, len(data), config.train.batch_size):
        indexes = range(start, min(start + config.train.batch_size, len(data)))
        batch = move_batch(
            collate_structure_examples([data[i] for i in indexes], pad_id=data.tokenizer.pad_id),
            "cuda",
        )
        prediction = model(batch)
        labels = batch["target_polar_token_ids"][:, 1:]
        mask = labels != data.tokenizer.pad_id
        eos = labels == data.tokenizer.eos_id
        tokens = prediction["polar_logits"].argmax(-1)
        polar_ce = F.cross_entropy(
            prediction["polar_logits"].transpose(1, 2),
            labels,
            ignore_index=data.tokenizer.pad_id,
            reduction="none",
        )
        chain_ce = F.cross_entropy(
            prediction["chain_logits"].transpose(1, 2),
            batch["target_chain_tokens"],
            reduction="none",
        )
        correct_chain = prediction["chain_logits"].argmax(-1) == batch["target_chain_tokens"]
        for j, index in enumerate(indexes):
            teacher[int(all_rows[index])] = {
                "tokens": int(mask[j].sum()),
                "correct": int(((tokens[j] == labels[j]) & mask[j]).sum()),
                "eos_correct": int(((tokens[j] == labels[j]) & eos[j]).sum()),
                "polar_loss_sum": float(polar_ce[j].sum()),
                "chain_loss_sum": float(chain_ce[j].sum()),
                "chain_slots_correct": int(correct_chain[j].sum()),
                "chain_exact": int(correct_chain[j].all()),
            }
        progress(
            {"stage": "teacher_forcing", "targets": min(start + config.train.batch_size, len(data))}
        )
    chains, heads = load_builder_rules(args.prepared, args.library, progress)
    index_for_row = {int(row): i for i, row in enumerate(all_rows)}
    generation_rows = np.concatenate([cohorts["smoke_128"], cohorts["diagnostic_512"]])
    records = []
    private_path = output / "raw_predictions_private.jsonl"
    with private_path.open("w") as private:
        for count, row in enumerate(generation_rows, 1):
            example = data[index_for_row[int(row)]]
            batch = move_batch(
                collate_structure_examples([example], pad_id=data.tokenizer.pad_id), "cuda"
            )
            greedy = model.generate(batch, beam_size=1, max_length=config.model.max_polar_length)
            beam = model.generate(batch, beam_size=5, max_length=config.model.max_polar_length)
            predicted = greedy["chain_logits"].argmax(-1)[0].cpu().tolist()
            true = example["target_chain_tokens"]
            chain_tokens = [data.chain_vocab.tokens[i] for i in predicted]
            target_polar = canonical_polar(example["polar_fragment_smiles"])
            template_polars = {canonical_polar(s) for s in example["template_polar_smiles"]}

            def score(raw):
                text = data.tokenizer.decode(raw["ids"])
                result = score_candidate(
                    text,
                    raw["eos"],
                    chain_tokens,
                    chains,
                    heads,
                    example["target_formula"].numpy(),
                    target_polar,
                    example["canonical_full_smiles"],
                    template_polars,
                )
                return {
                    **result,
                    "text": text,
                    "raw_ids": raw["ids"],
                    "log_probability": raw["log_probability"],
                }

            raw = score(greedy["polar_candidates"][0][0])
            beams = [score(c) for c in beam["polar_candidates"][0]]
            surviving = [c for c in beams if c["formula_match"]]
            record = {
                "row": int(row),
                "target_id": example["target_id"],
                "raw": raw,
                "beams": beams,
                "generated_length": len(raw["raw_ids"]) - 1,
                "top5_polar_recalled": any(c["polar_exact"] for c in beams),
                "chain_count_correct": sum(c != 0 for c in predicted) == sum(c != 0 for c in true),
                "chain_set_exact": sorted(predicted) == sorted(true),
                "nonnull_correct": sum(p == t for p, t in zip(predicted, true) if t != 0),
                "nonnull_total": sum(t != 0 for t in true),
                "predicted_chain_tokens": chain_tokens,
                "post_filter_selected": surviving[0] if surviving else None,
            }
            records.append(record)
            private.write(json.dumps(record) + "\n")
            private.flush()
            progress({"stage": "free_generation", "targets": count, "total": len(generation_rows)})
    output_cohorts = {}
    for name, rows in cohorts.items():
        selected_teacher = [teacher[int(r)] for r in rows]
        nt = sum(r["tokens"] for r in selected_teacher)
        teacher_metrics = {
            "evaluated_targets": len(rows),
            "evaluated_tokens": nt,
            "polar_token_accuracy": sum(r["correct"] for r in selected_teacher) / nt,
            "teacher_forced_eos_accuracy": sum(r["eos_correct"] for r in selected_teacher)
            / len(rows),
            "polar_loss": sum(r["polar_loss_sum"] for r in selected_teacher) / nt,
            "chain_loss": sum(r["chain_loss_sum"] for r in selected_teacher) / (2 * len(rows)),
            "chain_slot_accuracy": sum(r["chain_slots_correct"] for r in selected_teacher)
            / (2 * len(rows)),
            "chain_exact_accuracy": sum(r["chain_exact"] for r in selected_teacher) / len(rows),
        }
        teacher_metrics["loss"] = (
            config.train.lambda_chain * teacher_metrics["chain_loss"]
            + config.train.lambda_polar * teacher_metrics["polar_loss"]
        )
        output_cohorts[name] = {
            "selection": manifest["cohorts"][name],
            "teacher_forced": teacher_metrics,
        }
        if name != "validation_pool":
            members = set(rows.tolist())
            output_cohorts[name]["raw_generation"] = aggregate(
                [r for r in records if r["row"] in members]
            )
    torch.cuda.synchronize()
    result = {
        "schema": "midterm-quality-v1",
        "encoder": config.model.encoder,
        "step": checkpoint_step,
        "checkpoint_sha256": file_hash(args.checkpoint),
        "selection_manifest_sha256": file_hash(args.manifest),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "attention_backend": args.attention_backend,
        "precision": "fp32",
        "parameters": sum(p.numel() for p in model.parameters()),
        "cohorts": output_cohorts,
        "raw_outputs_preserved_privately": True,
        "forced_eos_or_repair": False,
        "training_or_test_samples_used": 0,
        "seconds": time.monotonic() - started,
        "peak_RAM_bytes": peak_ram_bytes(),
        "peak_VRAM_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    (output / "quality_report.json").write_text(json.dumps(result, indent=2))
    progress({"stage": "completed", "step": result["step"], "encoder": result["encoder"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default="outputs/midterm_v03/evaluation_manifest.json")
    p.add_argument("--prepared", default="data/private/pretraining_v03")
    p.add_argument("--frozen-index", default="data/private/morgan_index_v0.3")
    p.add_argument("--library", default="data/structure_library_v0.3")
    p.add_argument("--output", required=True)
    p.add_argument("--attention-backend", choices=["math"], default="math")
    args = p.parse_args()
    with attention_backend(args.attention_backend):
        evaluate(args)


if __name__ == "__main__":
    main()
