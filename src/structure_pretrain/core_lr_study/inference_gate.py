"""Validation-only reachability and real Beam32 probes, with no test access."""

import numpy as np
import pyarrow.parquet as pq
import torch

from structure_pretrain.core_ablation.beam import generate
from structure_pretrain.core_ablation.grammar import Budget, State, step
from structure_pretrain.core_lr_study.common import BASE_ROOT, original_run, read, write
from structure_pretrain.core_lr_study.data import StudyData, make_model


def audit(data, indices, event):
    budget = Budget(data.tokenizer, data.formulas[indices].cpu().numpy())
    count = 0
    for q, index in enumerate(indices):
        state, used = State(), np.zeros(11, dtype=np.int16)
        ids = data.tokens[index, 1 : int(data.lengths[index]) - 1].cpu().numpy()
        for token in ids:
            assert budget.countable[token]
            used += budget.increments[token]
            assert np.all(used <= budget.target[q])
            state = step(state, data.tokenizer.tokens[token])
            assert state is not None
        assert np.array_equal(used, budget.target[q]) and state.complete
        count += len(ids)
        if q % 512 == 0:
            event(stage="validation_gold_reachability", targets=q + 1, total=len(indices))
    return {
        "validation_targets": len(indices),
        "legal_prefixes_checked": count,
        "legal_targets_pruned": 0,
        "test_access": False,
    }


def run_inference_gate(root, device, event):
    data = StudyData(root, device)
    indices = np.load(root / "data/validation_indices.npy")
    assert len(indices) == 9493 and np.all(data.splits[indices] == 1)
    if device == "cpu":
        targets = pq.read_table(root / "data/validation_targets_private.parquet").to_pylist()
        assert len(targets) == len(indices)
        for index, target in zip(indices, targets):
            expected = data.tokenizer.encode(target["target_smiles"])
            assert data.tokens[index, : int(data.lengths[index])].tolist() == expected
            assert target["split"] == "validation"
        result = audit(data, indices, event)
        result["validation_table_token_alignment_verified"] = True
        write(root / "validation_reachability.json", result)
        return result
    model = make_model(root).to(device).eval()
    saved = torch.load(
        read(original_run() / "best.json")["path"], map_location=device, weights_only=False
    )
    assert saved["contract"] == read(BASE_ROOT / "queue_started.json")["contract"]
    model.load_state_dict(saved["model"])
    del saved
    ordered = indices[np.argsort(data.lengths[indices], kind="stable")]
    nonP = indices[data.formulas[indices, 4].cpu().numpy() == 0]
    selected = np.concatenate((ordered[-4:], nonP[:4]))
    inputs = data.batch(selected)
    del inputs["target_polar_token_ids"]
    with torch.inference_mode():
        encoded = model.encode_templates(inputs)
        result = generate(
            model.polar_decoder,
            encoded["template_memory"],
            encoded["memory_padding_mask"],
            data.tokenizer,
            inputs["target_core_formula"].cpu().numpy(),
            mode="elements_grammar",
            beam_size=32,
            max_length=96,
            progress=lambda t: event(stage="Beam32_probe", token=t),
        )
    budget = Budget(data.tokenizer, inputs["target_core_formula"].cpu().numpy())
    completed = 0
    for q, beams in enumerate(result):
        assert len(beams) <= 32
        for beam in beams:
            state, used = State(), np.zeros(11, dtype=np.int16)
            for token in beam["ids"][1:]:
                if token == data.tokenizer.eos_id:
                    assert np.array_equal(used, budget.target[q]) and state.complete
                    completed += 1
                    break
                used += budget.increments[token]
                assert np.all(used <= budget.target[q])
                state = step(state, data.tokenizer.tokens[token])
                assert state is not None
    assert completed > 0
    return {
        "validation_probe_targets": len(selected),
        "beam_size": 32,
        "EOS_candidates_checked": completed,
        "elements_and_grammar_replayed": True,
        "test_access": False,
    }
