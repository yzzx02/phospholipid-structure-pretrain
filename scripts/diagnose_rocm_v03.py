"""Small synchronized real-data diagnostic; host watchdog must supervise this process."""

import argparse
import faulthandler
import json
import os
import time
from pathlib import Path

import torch

from scripts.overfit_pretraining_v03 import select_rows
from structure_pretrain.config import ExperimentConfig
from structure_pretrain.data.batching import collate_structure_examples
from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.data.real_structure_dataset import RealStructureDataset
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel
from structure_pretrain.training.losses import pretraining_loss
from structure_pretrain.utils.attention import attention_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("fingerprint", "graph"), required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--attention-backend", choices=("math", "default"), default="math")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 50:
        parser.error("Only 1..50 synchronized diagnostic steps are allowed")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostic.pid").write_text(str(os.getpid()))
    started = time.monotonic()
    trace = (output / "traceback.log").open("w")
    faulthandler.enable(file=trace)
    faulthandler.dump_traceback_later(12, file=trace)

    def event(stage, **details):
        record = {"stage": stage, "elapsed_seconds": time.monotonic() - started, **details}
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(12, file=trace)

    event("gpu_probe")
    if not torch.version.hip or not torch.cuda.is_available():
        raise SystemExit("Working HIP GPU required")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    prepared = Path("data/private/pretraining_v03")
    selected = select_rows(prepared, 32, 42)[:2]
    data = RealStructureDataset(
        prepared,
        "data/private/morgan_index_v0.3",
        "train",
        args.encoder,
        "polar_excluded",
        selected,
    )
    examples = [data[i] for i in range(len(data))]
    batch = collate_structure_examples(examples, pad_id=data.tokenizer.pad_id)
    config = ExperimentConfig()
    config.model.encoder = args.encoder
    config.model.max_polar_length = data.max_length
    config.model.dropout = config.model.template_dropout = 0.0
    stats = read_json(prepared / "formula_stats_train_only.json")
    losses = []
    event("model_to_gpu", label_lengths=[len(e["target_polar_token_ids"]) for e in examples])
    with attention_backend(args.attention_backend):
        model = StructurePretrainModel(
            config.model, len(data.chain_vocab), data.tokenizer, stats
        ).to("cuda")
        batch = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in batch.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.chain_head.mlp[-1].weight.detach().clone()
        torch.cuda.synchronize()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            event("forward_started", step=step)
            loss = pretraining_loss(model(batch), batch, data.tokenizer.pad_id, config.train)[
                "loss"
            ]
            torch.cuda.synchronize()
            event("backward_started", step=step, loss=float(loss.detach()))
            loss.backward()
            torch.cuda.synchronize()
            event("optimizer_started", step=step)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite diagnostic loss")
            losses.append(float(loss.detach()))
            event("step_complete", step=step, loss=losses[-1], gradient_norm=float(norm))
            time.sleep(0.1)
        changed = not torch.equal(before, model.chain_head.mlp[-1].weight)
        if not changed:
            raise ValueError("Diagnostic parameters did not update")
    report = {
        "status": "passed",
        "scope": "bounded operator diagnosis, not overfit acceptance",
        "encoder": args.encoder,
        "steps": args.steps,
        "batch_size": 2,
        "real_train_targets": len(data),
        "attention_backend": args.attention_backend,
        "parameter_update": changed,
        "losses": losses,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "seconds": time.monotonic() - started,
    }
    (output / "diagnostic_report.json").write_text(json.dumps(report, indent=2))
    event("completed", status="passed")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
