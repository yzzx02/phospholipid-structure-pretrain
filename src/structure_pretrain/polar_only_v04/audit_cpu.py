"""Real frozen-data integration, executed before any v0.4 GPU job."""

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from structure_pretrain.data.prepare_v03 import read_json
from structure_pretrain.polar_only_v04.dataset import PolarDataset, collate_polar
from structure_pretrain.polar_only_v04.model import polar_loss
from structure_pretrain.polar_only_v04.prepare import DEFAULT_ROOT, SAMPLES, dump
from structure_pretrain.polar_only_v04.runtime import load_config, make_model
from structure_pretrain.utils.attention import attention_backend


def main():
    torch.set_num_threads(4)
    root = DEFAULT_ROOT
    assert read_json(root / "formula_audit.json")["status"] == "passed"
    assert read_json(root / "fragment_integrity_audit.json")["status"] == "passed"
    assert read_json(root / "retrieval_audit.json")["status"] == "passed"
    rows = np.load(SAMPLES / "overfit_32.npy")
    checks = {}
    for encoder, name in (("fingerprint", "fp"), ("graph", "gine")):
        config = load_config(f"configs/polar_only_v04_{name}.yaml", "overfit")
        data = PolarDataset(root, rows, encoder, force_h=True)
        batch = collate_polar([data[0], data[1]])
        model = make_model(config, data.tokenizer, root)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        before = {name: next(m.parameters()).detach().clone() for name, m in model.named_children()}
        with attention_backend("math"):
            output = model(batch)
            loss = polar_loss(output, batch, data.tokenizer.pad_id)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            model.eval()
            clean = {k: v for k, v in batch.items() if k != "target_polar_token_ids"}
            generated = model.generate(clean, max_length=8)
        updates = {
            name: not torch.equal(value, next(dict(model.named_children())[name].parameters()))
            for name, value in before.items()
        }
        assert all(updates.values())
        assert output["template_memory"].shape == (2, 4, 512)
        assert len(generated["polar_candidates"]) == 2
        # Exercise independent worker loading and bounded cache path on real data.
        loader = DataLoader(
            data, batch_size=2, num_workers=2, collate_fn=partial(collate_polar, k=3)
        )
        assert sum(len(b["target_full_formula"]) for b in loader) == 32
        checks[encoder] = {
            "finite_loss": float(loss.detach()),
            "parameters_changed": updates,
            "memory_shape": list(output["template_memory"].shape),
            "label_free_greedy": True,
            "worker_loaded_real_rows": 32,
        }
        print(json.dumps({"encoder": encoder, "CPU": "passed"}), flush=True)
    dump(
        Path("outputs/pretraining_v04/cpu_integration_audit.json"),
        {
            "status": "passed",
            "device": "cpu",
            "real_data": True,
            "encoders": checks,
            "pre_threshold_legal_reference_counts": pre_threshold_coverage(data),
        },
    )


def pre_threshold_coverage(data):
    """Count eligible TRAIN references before similarity threshold; no molecules allocated."""
    code = data.chains[:, 0].astype(np.int64) * 323 + data.chains[:, 1]
    train = np.flatnonzero(data.splits == 0)
    order = train[np.lexsort((data.groups[train], code[train]))]
    assert not np.any(
        (code[order[1:]] == code[order[:-1]]) & (data.groups[order[1:]] == data.groups[order[:-1]])
    )
    combos = np.bincount(code[train], minlength=323**2)
    heads = np.bincount(data.groups[train], minlength=int(data.groups.max()) + 1)
    is_train = data.splits == 0
    h = combos[code] - is_train
    hc = len(train) - combos[code] - heads[data.groups] + is_train
    result = {}
    for s, name in enumerate(("train", "validation", "test")):
        mask = data.splits == s
        result[name] = {
            mode: {
                "at_least": {str(k): int((counts[mask] >= k).sum()) for k in (1, 3, 5)},
                "zero": int((counts[mask] == 0).sum()),
            }
            for mode, counts in (("H", h), ("HC", hc))
        }
    return result


if __name__ == "__main__":
    main()
