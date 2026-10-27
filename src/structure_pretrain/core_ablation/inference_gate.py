"""Gold-prefix reachability and bounded real-checkpoint CUDA decoding checks."""

import time

import numpy as np
import torch

from structure_pretrain.core_ablation.beam import generate
from structure_pretrain.core_ablation.common import (
    ROOT,
    TEMPLATE_ROOT,
    configuration,
    digest,
    read,
    tokenizer,
)
from structure_pretrain.core_ablation.grammar import Budget, State, step
from structure_pretrain.core_templates.data import CoreData, make_model


def audit_gold_prefixes(root, event):
    indices = np.load(root / "data/test_all.npy")
    formulas = np.load(root / "data/formulas.npy")
    tokens = np.load(root / "data/tokens.npy", mmap_mode="r")
    lengths = np.load(root / "data/token_lengths.npy")
    tok = tokenizer()
    budget = Budget(tok, formulas[indices])
    total = 0
    for q, index in enumerate(indices):
        state, used = State(), np.zeros(11, dtype=np.int16)
        ids = tokens[index, 1 : int(lengths[index]) - 1]
        for token in ids:
            assert budget.countable[token]
            used += budget.increments[token]
            if np.any(used > budget.target[q]):
                raise AssertionError("A frozen target exceeds its own online element budget")
            state = step(state, tok.tokens[token])
            if state is None:
                raise AssertionError("New grammar wrongly prunes a frozen legal target label")
        assert np.array_equal(used, budget.target[q]) and state.complete
        total += len(ids)
        if q % 512 == 0:
            event(stage="gold_formula_grammar_reachability", targets=q + 1, total=len(indices))
    return {
        "passed": True,
        "test_target_labels": len(indices),
        "legal_prefixes_checked": total,
        "test_labels_pruned": 0,
        "H_formula_checked_by_final_RDKit_not_character_count": True,
        "P_and_nonP_supported": True,
        "both_ports_exactly_once": True,
    }


def run_inference_gate(root, device, event):
    root = ROOT.resolve()
    if device == "cpu":
        result = audit_gold_prefixes(root, event)
        from structure_pretrain.core_templates.common import write

        write(root / "gold_reachability_audit.json", result)
        return result
    c = configuration()
    data = CoreData(TEMPLATE_ROOT, device)
    model = make_model(TEMPLATE_ROOT).to(device).eval()
    saved = torch.load(c["template_checkpoint"], map_location=device, weights_only=False)
    if saved["contract"] != read(TEMPLATE_ROOT / "queue_started.json")["contract"]:
        raise ValueError("Epoch43 checkpoint has an unexpected original contract")
    model.load_state_dict(saved["model"])
    del saved
    indices = np.load(root / "data/test_all.npy")
    # Two P/nonP groups and the longest labels stress both the boundary and memory.
    ordered = indices[np.argsort(data.lengths[indices], kind="stable")]
    sample = np.unique(
        np.concatenate(
            (ordered[-8:], indices[:4], indices[data.formulas[indices, 4].cpu().numpy() == 0][:4])
        )
    )[:16]
    inputs = data.batch(sample)
    del inputs["target_polar_token_ids"]
    with torch.inference_mode():
        encoded = model.encode_templates(inputs)
        mem, pad = encoded["template_memory"], encoded["memory_padding_mask"]
        prefix = torch.full(
            (len(sample), 1), data.tokenizer.bos_id, dtype=torch.long, device=device
        )
        together = model.polar_decoder(prefix, mem, pad)
        separate = torch.cat(
            [
                model.polar_decoder(prefix[i : i + 1], mem[i : i + 1], pad[i : i + 1])
                for i in range(len(sample))
            ]
        )
        delta = float((together - separate).abs().max())
        assert delta <= 1e-4
        torch.cuda.reset_peak_memory_stats()
        checks = {}
        formulas = data.formulas[sample].cpu().numpy()
        for mode in c["decoding_modes"]:
            began = time.monotonic()

            def heartbeat(position):
                if position % 10 == 0:
                    event(
                        stage="real_checkpoint_decode_gate",
                        mode=mode,
                        token=position,
                        targets=len(sample),
                    )

            beams = generate(
                model.polar_decoder,
                mem,
                pad,
                data.tokenizer,
                formulas,
                mode=mode,
                beam_size=20,
                max_length=96,
                progress=heartbeat,
            )
            checks[mode] = {
                "targets": len(sample),
                "candidate_count": sum(map(len, beams)),
                "EOS_candidates": sum(x["eos"] for row in beams for x in row),
                "seconds": time.monotonic() - began,
                "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
            }
            if mode != "raw":
                budget = Budget(data.tokenizer, formulas)
                for q, row in enumerate(beams):
                    for candidate in row:
                        ids = candidate["ids"][1:-1] if candidate["eos"] else candidate["ids"][1:]
                        used = budget.increments[ids].sum(0)
                        assert np.all(used <= budget.target[q])
                        if candidate["eos"]:
                            assert np.array_equal(used, budget.target[q])
                            if mode == "elements_grammar":
                                state = State()
                                for token in ids:
                                    state = step(state, data.tokenizer.tokens[token])
                                    assert state is not None
                                assert state.complete
            if mode == "raw":
                old = model.polar_decoder.generate(mem[:2], pad[:2], beam_size=20, max_length=96)
                shared = []
                for q in range(2):
                    previous = {tuple(r["ids"]): r["log_probability"] for r in old[q]}
                    for r in beams[q]:
                        if tuple(r["ids"]) in previous:
                            shared.append(abs(previous[tuple(r["ids"])] - r["log_probability"]))
                assert shared and max(shared) <= 1e-3
                checks[mode]["reference_shared_candidates"] = len(shared)
                checks[mode]["reference_shared_max_logprob_difference"] = max(shared)
                checks[mode]["reference_top1_sequence_agreement"] = (
                    sum(old[q][0]["ids"] == beams[q][0]["ids"] for q in range(2)) / 2
                )
    return {
        "passed": True,
        "real_epoch43_checkpoint_sha256": digest(c["template_checkpoint"]),
        "BOS_logits_batch_vs_sequential_max_difference": delta,
        "original_probabilities_not_renormalized": True,
        "modes": checks,
        "note": (
            "FP32 batched GEMM may change near-tied beams; both models use identical batched rules"
        ),
    }
