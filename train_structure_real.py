"""Explicit bounded real v0.3 training; smoke never transitions to full training."""

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler

from structure_pretrain.config import ExperimentConfig
from structure_pretrain.data.batching import collate_structure_examples
from structure_pretrain.data.prepare_v03 import SCHEMA, canonical_polar, read_json, rule_from_row
from structure_pretrain.data.real_structure_dataset import RealStructureDataset, sample_targets
from structure_pretrain.library_builder.pipeline import file_hash
from structure_pretrain.library_builder.schemas import ChainRule, PolarRule
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel
from structure_pretrain.polar_plausibility.pipeline import dump_json
from structure_pretrain.training.checkpoint_validation import validate_reload
from structure_pretrain.training.generation_metrics import evaluate_generation
from structure_pretrain.training.losses import pretraining_loss
from structure_pretrain.utils.attention import attention_backend


class ReproducibleBatchSampler(Sampler):
    """Checkpoint commits consumed samples, unaffected by DataLoader prefetch."""

    def __init__(self, size, batch_size, batches, seed, consumed=0):
        self.size, self.batch_size, self.batches = size, batch_size, batches
        self.seed, self.consumed = seed, consumed

    def __len__(self):
        return self.batches

    def __iter__(self):
        position, epoch, order = self.consumed, None, None
        for _ in range(self.batches):
            batch = []
            while len(batch) < self.batch_size:
                current, offset = divmod(position, self.size)
                if epoch != current:
                    order = np.random.default_rng(self.seed + current).permutation(self.size)
                    epoch = current
                take = min(self.size - offset, self.batch_size - len(batch))
                batch.extend(order[offset : offset + take].tolist())
                position += take
            yield batch


def move_batch(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def peak_ram_bytes():
    import sys

    if sys.platform != "win32":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
    try:
        import psutil

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss)
    except ImportError:
        raise RuntimeError("psutil is required for Windows peak RSS measurement") from None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/pretraining_v03_local_fp.yaml")
    p.add_argument("--prepared", default="data/private/pretraining_v03")
    p.add_argument("--frozen-index", default="data/private/morgan_index_v0.3")
    p.add_argument("--library", default="data/structure_library_v0.3")
    p.add_argument("--output", default="outputs/pretraining_v03/local_smoke/fingerprint")
    p.add_argument("--device", choices=("cpu", "rocm"), default="cpu")
    p.add_argument("--steps", type=int)
    p.add_argument("--resume")
    p.add_argument("--allow-full-training", action="store_true")
    p.add_argument("--gpu-smoke-root", default="outputs/pretraining_v03/rocm_smoke")
    p.add_argument("--attention-backend", choices=("default", "math"), default="default")
    args = p.parse_args()
    with attention_backend(args.attention_backend):
        execute(args, p)


def execute(args, p):
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(ExperimentConfig()),
            {k: v for k, v in raw.items() if k in ("model", "train", "retrieval")},
        )
    )
    run = raw["run"]
    if (config.retrieval.radius, config.retrieval.fp_size, config.retrieval.minimum_similarity) != (
        2,
        2048,
        0.4,
    ):
        p.error("Frozen v0.3 retrieval requires Morgan radius2/2048/threshold0.4")
    if not 1 <= config.retrieval.max_top_k <= 10:
        p.error("max_top_k must be 1..10")
    if run["kind"] == "rocm_smoke" and args.device != "rocm":
        p.error("The ROCm smoke config requires --device rocm and a passing device check")
    steps = args.steps if args.steps is not None else run["steps"]
    if steps < 1:
        p.error("steps must be positive")
    full = run["kind"] == "full"
    if full and not args.allow_full_training:
        p.error("Full training requires its separate config AND --allow-full-training")
    if full:
        if args.device != "rocm":
            p.error("This full config is reserved for the verified ROCm host")
        for name in ("fingerprint", "graph"):
            smoke = Path(args.gpu_smoke_root) / name
            metrics = read_json(smoke / "training_metrics.json")
            resources = read_json(smoke / "resource_metrics.json")
            if (
                not resources.get("hip")
                or not resources.get("device", "").startswith("cuda")
                or metrics["optimizer_steps_total"] < 1000
                or metrics["train_targets"] < 20000
                or metrics["validation_targets"] < 2000
                or not all(metrics["parameters_changed"].values())
                or not metrics.get(
                    "checkpoint_reload_logits_close", metrics["checkpoint_reload_identical_logits"]
                )
            ):
                p.error(f"Real GPU smoke gate failed for {name}")
    if not full and steps > run["step_limit"]:
        p.error("Smoke step limit exceeded; smoke cannot silently become full training")
    if args.device == "rocm":
        if not torch.version.hip or not torch.cuda.is_available():
            p.error("Run scripts/check_rocm.py first: HIP PyTorch/available GPU required")
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
    else:
        device = torch.device("cpu")
        if run["precision"] != "fp32":
            p.error("Local CPU smoke requires fp32")
    torch.set_num_threads(config.train.cpu_threads)
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    root, output = Path(args.prepared), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostic.pid").write_text(str(os.getpid()))

    def log_event(record):
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    log_event({"stage": "preparing", "attention_backend": args.attention_backend})
    label = read_json(root / "labels_manifest.json")
    if full:
        for name in ("fingerprint", "graph"):
            contract_check = read_json(Path(args.gpu_smoke_root) / name / "training_metrics.json")[
                "contract"
            ]
            if contract_check["structures_sha256"] != label["structures_sha256"]:
                raise ValueError("GPU smoke belongs to a different frozen library")
    for name in (
        "label_audit.json",
        "tokenization_audit.json",
        "reconstruction_audit.json",
        "split_audit.json",
        "retrieval_integrity_audit.json",
    ):
        if read_json(root / name)["status"] != "passed":
            raise ValueError(f"Training gate failed: {name}")
    # Configured label length can never silently truncate the audited data.
    config.model.max_polar_length = label["max_polar_length"]
    if not full:
        special_ids = pq.read_table(
            Path(args.library) / "special_cases.parquet", columns=["structure_id"]
        )["structure_id"].to_pylist()
        all_ids = np.load(Path(args.frozen_index) / "row_ids.npy", mmap_mode="r")
        priority_rows = np.flatnonzero(np.isin(all_ids, np.asarray(special_ids, dtype="S67")))
        samples = sample_targets(
            root,
            run["train_targets"],
            run["validation_targets"],
            config.train.seed,
            priority_rows,
            sample_output=output / "sample_rows",
        )
    else:
        splits = np.load(root / "row_splits.npy", mmap_mode="r")
        samples = {
            name: np.flatnonzero(splits == i) for i, name in enumerate(("train", "validation"))
        }
    train = RealStructureDataset(
        root,
        args.frozen_index,
        "train",
        config.model.encoder,
        run["retrieval_mode"],
        samples["train"],
        seed=config.train.seed,
        cross_fraction=run["cross_fraction"],
        max_top_k=config.retrieval.max_top_k,
    )
    validation = RealStructureDataset(
        root,
        args.frozen_index,
        "validation",
        config.model.encoder,
        "polar_excluded",
        samples["validation"],
        seed=config.train.seed,
        max_top_k=config.retrieval.max_top_k,
    )
    stats = read_json(root / "formula_stats_train_only.json")
    if stats["structures_sha256"] != label["structures_sha256"] or not stats["training_pairs_only"]:
        raise ValueError("Wrong formula standardizer provenance")
    contract = {
        "schema": SCHEMA,
        "version": "v0.3",
        "structures_sha256": label["structures_sha256"],
        "labels_manifest_sha256": file_hash(root / "labels_manifest.json"),
        "tokenizer_sha256": file_hash(root / "polar_tokenizer.json"),
        "chain_vocab_sha256": file_hash(root / "chain_vocab.json"),
        "split_sha256": file_hash(root / "row_splits.npy"),
        "formula_stats_sha256": file_hash(root / "formula_stats_train_only.json"),
        "retrieval_sha256": file_hash(root / "retrieval_coverage_by_split.json"),
        "model_config": asdict(config.model),
        "train_config": asdict(config.train),
        "retrieval_config": asdict(config.retrieval),
        "run": run,
        "train_targets_sha256": hashlib.sha256(samples["train"].tobytes()).hexdigest(),
        "validation_targets_sha256": hashlib.sha256(samples["validation"].tobytes()).hexdigest(),
        "outputs": ["canonical_chain_slots_2", "P_containing_polar_fragment"],
        "polar_boundary": "glycerol-O | P-containing polar fragment",
    }
    if args.attention_backend != "default":
        contract["attention_backend"] = args.attention_backend
    model = StructurePretrainModel(config.model, len(train.chain_vocab), train.tokenizer, stats).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    precision = run["precision"]
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    step, consumed, history = 0, 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("contract") != contract:
            raise ValueError(
                "Checkpoint/data/model/split/vocabulary contract mismatch; "
                "old checkpoints incompatible"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step, consumed, history = checkpoint["step"], checkpoint["consumed"], checkpoint["history"]
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        np.random.set_state(checkpoint["rng_numpy"])
        random.setstate(checkpoint["rng_python"])
        if device.type == "cuda" and checkpoint.get("rng_device") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng_device"]])
    checkpoint_dir = output / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "last.pt"

    def save_checkpoint():
        state = {
            "contract": contract,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "consumed": consumed,
            "history": history,
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
            "rng_device": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        torch.save(state, checkpoint_path.with_suffix(".partial"))
        checkpoint_path.with_suffix(".partial").replace(checkpoint_path)

    if steps <= step:
        p.error("--steps is total optimizer steps and must exceed the resume step")
    start_step = step
    collate = partial(
        collate_structure_examples,
        pad_id=train.tokenizer.pad_id,
        max_top_k=config.retrieval.max_top_k,
    )
    accum = run["gradient_accumulation"]
    loader_options = {"num_workers": config.train.num_workers, "collate_fn": collate}
    if config.train.num_workers:
        loader_options.update({"prefetch_factor": 2, "persistent_workers": True})
    sampler = ReproducibleBatchSampler(
        len(train), config.train.batch_size, (steps - step) * accum, config.train.seed, consumed
    )
    loader = DataLoader(
        train,
        batch_sampler=sampler,
        generator=torch.Generator().manual_seed(config.train.seed),
        **loader_options,
    )
    val_loader = DataLoader(
        validation,
        batch_size=config.train.batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.train.seed + 1),
        **loader_options,
    )
    resolved = {
        "experiment": asdict(config),
        "run": run,
        "requested_total_steps": steps,
        "device": str(device),
        "prepared": args.prepared,
        "frozen_index": args.frozen_index,
        "contract": contract,
    }
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True), encoding="utf-8"
    )
    watched = {name: next(module.parameters()) for name, module in model.named_children()}
    before = {name: value.detach().clone() for name, value in watched.items()}
    gradient_checks = {}
    started, loading_seconds, micro, loss_accum = time.monotonic(), 0.0, 0, CounterValues()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    previous = time.monotonic()
    for cpu_batch in loader:
        loading_seconds += time.monotonic() - previous
        batch = move_batch(cpu_batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
            prediction = model(batch)
            losses = pretraining_loss(prediction, batch, train.tokenizer.pad_id, config.train)
            loss = losses["loss"] / accum
        if not all(torch.isfinite(v) for v in losses.values()):
            raise ValueError("Nonfinite real training loss")
        scaler.scale(loss).backward()
        micro += 1
        consumed += len(cpu_batch["target_id"])
        loss_accum.add({k: float(v.detach()) / accum for k, v in losses.items()})
        if micro % accum == 0:
            scaler.unscale_(optimizer)
            for name, module in model.named_children():
                grads = [p.grad for p in module.parameters() if p.requires_grad]
                if any(g is None or not torch.isfinite(g).all() for g in grads):
                    raise ValueError(f"Missing/nonfinite gradient in {name}")
                gradient_checks[name] = any(g.abs().sum() > 0 for g in grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = {"step": step, **loss_accum.values}
            history.append(metrics)
            loss_accum = CounterValues()
            log_event(metrics)
            if step % run.get("checkpoint_every", 500) == 0:
                save_checkpoint()
        previous = time.monotonic()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    updates = {name: not torch.equal(value, before[name]) for name, value in watched.items()}
    if not all(updates.values()) or not all(gradient_checks.values()):
        raise ValueError("Some real model branches failed to update")
    model.eval()
    validation_totals, seen = CounterValues(), 0
    token_count = 0
    first_batch = None
    with torch.no_grad():
        for cpu_batch in val_loader:
            batch = move_batch(cpu_batch, device)
            if first_batch is None:
                first_batch = batch
            prediction = model(batch)
            losses = pretraining_loss(prediction, batch, train.tokenizer.pad_id, config.train)
            n = len(batch["target_id"])
            labels = batch["target_polar_token_ids"][:, 1:]
            valid = labels.ne(train.tokenizer.pad_id)
            chain_correct = prediction["chain_logits"].argmax(-1).eq(batch["target_chain_tokens"])
            polar_correct = prediction["polar_logits"].argmax(-1).eq(labels) & valid
            nt = int(valid.sum())
            validation_totals.add(
                {
                    "chain_loss": float(losses["chain_loss"]) * n * 2,
                    "polar_loss": float(losses["polar_loss"]) * nt,
                    "chain_slot_correct": int(chain_correct.sum()),
                    "chain_exact_correct": int(chain_correct.all(-1).sum()),
                    "polar_token_correct": int(polar_correct.sum()),
                }
            )
            token_count += nt
            seen += n
            if seen % (config.train.batch_size * 10) == 0:
                log_event({"stage": "validation", "targets": seen})
    sums = validation_totals.values
    validation_metrics = {
        "chain_loss": sums["chain_loss"] / (2 * seen),
        "polar_loss": sums["polar_loss"] / token_count,
        "chain_slot_accuracy": sums["chain_slot_correct"] / (2 * seen),
        "chain_exact_accuracy": sums["chain_exact_correct"] / seen,
        "polar_token_accuracy": sums["polar_token_correct"] / token_count,
        "evaluated_targets": seen,
        "evaluated_tokens": token_count,
    }
    validation_metrics["loss"] = (
        config.train.lambda_chain * validation_metrics["chain_loss"]
        + config.train.lambda_polar * validation_metrics["polar_loss"]
    )
    save_checkpoint()
    restored = (
        StructurePretrainModel(config.model, len(train.chain_vocab), train.tokenizer, stats)
        .to(device)
        .eval()
    )
    restored.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)["model"]
    )
    reference, reload_report = validate_reload(
        model, restored, first_batch, allow_roundoff=device.type == "cuda"
    )
    del restored
    generated_count = min(run["generation_targets"], len(validation))
    generation_batches = (
        move_batch(collate([validation[i]]), device) for i in range(generated_count)
    )
    chain_rules = {
        r["chain_token"]: rule_from_row(ChainRule, r)
        for r in pq.read_table(Path(args.library) / "chain_library.parquet").to_pylist()
    }
    polar_rules = {}
    log_event({"stage": "loading_polar_rules", "heads": 0})
    for part in pq.ParquetFile(Path(args.library) / "polar_library.parquet").iter_batches(
        batch_size=2048
    ):
        for r in part.to_pylist():
            polar_rules[canonical_polar(r["polar_fragment_smiles"])] = rule_from_row(PolarRule, r)
        log_event({"stage": "loading_polar_rules", "heads": len(polar_rules)})
    generation, predictions = evaluate_generation(
        model,
        generation_batches,
        train.tokenizer,
        train.chain_vocab,
        chain_rules,
        polar_rules,
        beam_size=5,
        max_length=config.model.max_polar_length,
        progress=log_event,
    )
    training = {
        "schema": SCHEMA,
        "contract": contract,
        "optimizer_steps_total": step,
        "optimizer_steps_this_run": step - start_step,
        "resumed_from_step": start_step,
        "train_targets": len(train),
        "train_examples_consumed": consumed,
        "unique_train_targets_seen": min(consumed, len(train)),
        "validation_targets": len(validation),
        "test_targets_used_for_training_or_tuning": 0,
        "train_history": history,
        "validation_teacher_forced": validation_metrics,
        "gradient_checks": gradient_checks,
        "parameters_changed": updates,
        **reload_report,
        "parameters": sum(p.numel() for p in model.parameters()),
        "module_parameters": {
            n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children()
        },
        "shapes": {
            k: list(reference[k].shape) for k in ("chain_logits", "polar_logits", "template_tokens")
        },
        "scientific_scope": "real-data engineering smoke under structure-oracle retrieval",
    }
    resource = {
        "device": str(device),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "attention_backend": args.attention_backend,
        "precision": precision,
        "optimizer_training_seconds": elapsed,
        "steps_per_second": (step - start_step) / elapsed,
        "samples_per_second": (step - start_step) * config.train.batch_size * accum / elapsed,
        "data_loading_seconds": loading_seconds,
        "data_loading_samples_per_second": (step - start_step)
        * config.train.batch_size
        * accum
        / max(loading_seconds, 1e-9),
        "peak_RAM_bytes": peak_ram_bytes(),
        "peak_RAM_scope": "main process peak resident memory",
        "data_loader_workers_per_loader": config.train.num_workers,
        "peak_VRAM_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else None,
        "peak_VRAM_reserved_bytes": torch.cuda.max_memory_reserved()
        if device.type == "cuda"
        else None,
    }
    dump_json(output / "training_metrics.json", training)
    dump_json(output / "generation_metrics.json", generation)
    dump_json(output / "resource_metrics.json", resource)
    dump_json(output / "sample_predictions_private.json", predictions)
    print(
        json.dumps(
            {
                "completed_steps": step,
                "validation": validation_metrics,
                "generation": generation,
                "resources": resource,
            }
        ),
        flush=True,
    )


class CounterValues:
    def __init__(self):
        self.values = {}

    def add(self, values):
        for key, value in values.items():
            self.values[key] = self.values.get(key, 0.0) + value


if __name__ == "__main__":
    main()
