"""Hardware preflight only: real GINE/Transformer operations, AMP, optimizer and resume."""

import argparse
import json
from pathlib import Path

import torch

from structure_pretrain.config import ExperimentConfig
from structure_pretrain.data.batching import collate_structure_examples
from structure_pretrain.data.dummy_structure_dataset import DummyStructureDataset
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel
from structure_pretrain.training.losses import pretraining_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/pretraining_v03/rocm_preflight")
    args = parser.parse_args()
    info = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu_available": torch.cuda.is_available(),
    }
    print(json.dumps(info, indent=2))
    if not torch.version.hip or not torch.cuda.is_available():
        raise SystemExit("Install AMD's matching WSL ROCm PyTorch wheels before training")
    device = torch.device("cuda")  # ROCm intentionally exposes PyTorch's cuda API.
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for encoder in ("fingerprint", "graph"):
        config = ExperimentConfig()
        config.model.encoder = encoder
        data = DummyStructureDataset(encoder)
        batch = collate_structure_examples([data[0], data[1]])
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        for precision, dtype in (
            ("fp32", torch.float32),
            ("bf16", torch.bfloat16),
            ("fp16", torch.float16),
        ):
            model = StructurePretrainModel(config.model, len(data.chain_vocab), data.tokenizer).to(
                device
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=128.0)
            before = model.chain_head.mlp[-1].weight.detach().clone()
            with torch.autocast("cuda", dtype=dtype, enabled=precision != "fp32"):
                loss = pretraining_loss(model(batch), batch, data.tokenizer.pad_id, config.train)[
                    "loss"
                ]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite = bool(
                torch.isfinite(loss)
                and all(
                    p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
                )
            )
            if not finite:
                raise SystemExit(f"ROCm {encoder}/{precision}: nonfinite forward/backward")
            scaler.step(optimizer)
            scaler.update()
            updated = not torch.equal(before, model.chain_head.mlp[-1].weight)
            checkpoint = output / f"{encoder}_{precision}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                checkpoint,
            )
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            optimizer.zero_grad(set_to_none=True)
            resumed = pretraining_loss(model(batch), batch, data.tokenizer.pad_id, config.train)[
                "loss"
            ]
            resumed.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if not updated or not torch.isfinite(resumed):
                raise SystemExit("ROCm optimizer/checkpoint resume failed")
            results.append(
                {
                    "encoder": encoder,
                    "precision": precision,
                    "status": "passed",
                    "loss": float(loss.detach()),
                    "resumed_loss": float(resumed.detach()),
                    "parameter_update": updated,
                    "checkpoint_resume": True,
                }
            )
            del model, optimizer, scaler
    report = {
        **info,
        "status": "passed",
        "gpu": torch.cuda.get_device_name(),
        "checks": results,
        "scope": "hardware operator check with dummy fixtures; real-data GPU smoke still required",
    }
    (output / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
