"""Run structure-only dummy training on CPU or an explicitly verified ROCm device."""

import argparse
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from structure_pretrain.config import load_config
from structure_pretrain.data.batching import collate_structure_examples
from structure_pretrain.data.dummy_structure_dataset import DummyStructureDataset
from structure_pretrain.training.lightning_module import StructurePretrainLightningModule
from structure_pretrain.utils.formula import FormulaStandardizer
from structure_pretrain.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/structure_pretrain_graph.yaml")
    parser.add_argument("--device", choices=("cpu", "rocm", "auto"), default="cpu")
    parser.add_argument(
        "--precision", choices=("32-true", "16-mixed", "bf16-mixed"), default="32-true"
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", default="outputs/dummy")
    parser.add_argument(
        "--resume", help="Restore model, optimizer and Trainer from this checkpoint"
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.device == "rocm" and (not torch.version.hip or not torch.cuda.is_available()):
        parser.error(
            "ROCm requested but HIP PyTorch/available GPU missing; run scripts/check_rocm.py"
        )
    use_gpu = args.device == "rocm" or (args.device == "auto" and torch.cuda.is_available())
    if not use_gpu and args.precision != "32-true":
        parser.error("CPU smoke verification uses --precision 32-true")
    config = load_config(args.config)
    seed_everything(config.train.seed, config.train.cpu_threads)
    dataset = DummyStructureDataset(config.model.encoder, config.retrieval)
    if len(dataset) < 8:
        raise RuntimeError("Expected eight nonempty dummy targets")
    # Statistics use ONLY training retrieval pairs; validation examples are held out.
    train_examples, val_examples = dataset.examples[:6], dataset.examples[6:]
    statistics = FormulaStandardizer.fit(train_examples)
    module = StructurePretrainLightningModule(
        asdict(config), list(dataset.chain_vocab.tokens), statistics
    )
    if args.resume:
        restored = StructurePretrainLightningModule.load_from_checkpoint(
            args.resume, map_location="cpu"
        )
        if asdict(restored.config) != asdict(config):
            raise ValueError("Resume requires the original configuration")
        module = restored
    collate = partial(
        collate_structure_examples,
        max_top_k=config.retrieval.max_top_k,
        pad_id=dataset.tokenizer.pad_id,
    )
    loader_options = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "collate_fn": collate,
        "shuffle": False,
    }
    train_loader = DataLoader(train_examples, **loader_options)
    val_loader = DataLoader(val_examples, **loader_options)
    output_dir = Path(args.output)
    trainer = L.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=args.precision,
        max_steps=args.steps,
        max_epochs=-1,
        logger=CSVLogger(str(output_dir), name=config.model.encoder),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        gradient_clip_val=config.train.gradient_clip,
        log_every_n_steps=1,
        default_root_dir=str(output_dir),
    )
    trainer.fit(
        module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume
    )
    validation = trainer.validate(module, dataloaders=val_loader, verbose=False)[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "last.ckpt"
    trainer.save_checkpoint(checkpoint)
    module.eval()
    batch = module.transfer_batch_to_device(collate(val_examples), module.device, 0)
    with torch.no_grad():
        prediction = module(batch)
    summary = {
        "schema": "structure-pretraining-v0.3-two-output-dummy",
        "encoder": config.model.encoder,
        "data_kind": "toy_structures_artificial_labels_not_real_phospholipid_training",
        "global_step": trainer.global_step,
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "structure_encoder_parameters": sum(
            parameter.numel() for parameter in module.model.structure_encoder.parameters()
        ),
        "shapes": {
            key: list(prediction[key].shape)
            for key in (
                "chain_logits",
                "polar_logits",
                "template_tokens",
                "template_memory",
                "template_global",
            )
        },
        "validation": validation,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(module.device),
        "precision": args.precision,
        "checkpoint": str(checkpoint.resolve()),
        "train_targets": len(train_examples),
        "validation_targets": len(val_examples),
        "skipped_target_ids": dataset.skipped_target_ids,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
