"""Batched Beam20 with original mean-log-probability ranking and optional budgets."""

import numpy as np
import torch

from structure_pretrain.core_ablation.grammar import Budget, State
from structure_pretrain.core_ablation.grammar import step as syntax_step


@torch.inference_mode()
def generate(
    decoder,
    memory,
    padding,
    tokenizer,
    formulas,
    *,
    mode,
    beam_size=20,
    max_length=96,
    progress=None,
):
    if mode not in ("raw", "elements", "elements_grammar"):
        raise ValueError("Unknown decode mode")
    if memory.ndim != 3 or not 2 <= max_length <= decoder.max_length or beam_size < 1:
        raise ValueError("Invalid generation shape/limit")
    b, k, device = len(memory), beam_size, memory.device
    seq = torch.full((b, k, max_length), tokenizer.pad_id, dtype=torch.long, device=device)
    seq[:, :, 0] = tokenizer.bos_id
    scores = torch.full((b, k), -torch.inf, dtype=torch.float64, device=device)
    scores[:, 0] = 0
    lengths = torch.ones((b, k), dtype=torch.long, device=device)
    done = torch.zeros((b, k), dtype=torch.bool, device=device)
    budget = Budget(tokenizer, formulas) if mode != "raw" else None
    consumed = np.zeros((b, k, 11), dtype=np.int16)
    states = [[State()] + [None] * (k - 1) for _ in range(b)]
    for position in range(max_length - 1):
        active = torch.isfinite(scores) & ~done
        if not bool(active.any()):
            break
        active_at = active.nonzero()
        queries = active_at[:, 0]
        ids = seq[:, :, : position + 1][active]
        logits = decoder(ids, memory[queries], padding[queries])[:, -1].clone()
        logits[:, [tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]] = -torch.inf
        logp = logits.log_softmax(-1)
        if bool(torch.isnan(logp).any()) or bool(torch.isposinf(logp).any()):
            raise ValueError("Invalid decoder probabilities")
        if budget is not None:
            locations = active_at.cpu().numpy()
            q, j = locations[:, 0], locations[:, 1]
            syntax = [states[x][y] for x, y in locations] if mode == "elements_grammar" else None
            mask = budget.allowed(consumed[q, j], q, syntax)
            logp.masked_fill_(~torch.as_tensor(mask, device=device), -torch.inf)
        values, choices = logp.topk(k)
        top = torch.full((b, k, k), -torch.inf, dtype=torch.float64, device=device)
        tokens = torch.zeros((b, k, k), dtype=torch.long, device=device)
        top[active] = values.to(torch.float64)
        tokens[active] = choices
        # Stable ordering exactly matches: old finished beams first, then active
        # parents in beam order, then their original torch.topk token order.
        finished = scores.masked_fill(~done, -torch.inf)
        extension = scores[:, :, None] + top
        pool = torch.cat((finished, extension.flatten(1)), dim=1)
        pool_lengths = torch.cat(
            (lengths, torch.full((b, k * k), position + 2, device=device, dtype=torch.long)), dim=1
        )
        ranking = pool / (pool_lengths - 1).clamp_min(1)
        chosen = torch.argsort(ranking, descending=True, stable=True, dim=1)[:, :k]
        is_extension = chosen >= k
        offset = (chosen - k).clamp_min(0)
        parents = torch.where(is_extension, offset // k, chosen)
        next_tokens = tokens.flatten(1).gather(1, offset)
        next_scores = pool.gather(1, chosen)
        valid = torch.isfinite(next_scores)
        next_seq = seq.gather(1, parents[:, :, None].expand(-1, -1, max_length))
        extending = is_extension & valid
        next_seq[:, :, position + 1] = torch.where(
            extending, next_tokens, next_seq[:, :, position + 1]
        )
        next_lengths = torch.where(is_extension, position + 2, lengths.gather(1, parents))
        next_done = valid & (
            done.gather(1, parents) | (extending & next_tokens.eq(tokenizer.eos_id))
        )
        if budget is not None:
            p, t, ext, ok = (x.cpu().numpy() for x in (parents, next_tokens, is_extension, valid))
            consumed = np.take_along_axis(consumed, p[:, :, None], axis=1)
            consumed += budget.increments[t] * (ext & ok)[:, :, None]
            if mode == "elements_grammar":
                updated = []
                for q in range(b):
                    row = []
                    for j in range(k):
                        state = states[q][p[q, j]] if ok[q, j] else None
                        if ok[q, j] and ext[q, j] and t[q, j] != tokenizer.eos_id:
                            state = syntax_step(state, tokenizer.tokens[int(t[q, j])])
                            if state is None:
                                raise AssertionError("A masked grammar transition was selected")
                        row.append(state)
                    updated.append(row)
                states = updated
        seq, scores, lengths, done = next_seq, next_scores, next_lengths, next_done
        if progress:
            progress(position + 1)
    sequences, probs, sizes, finished = (x.cpu().numpy() for x in (seq, scores, lengths, done))
    result = []
    for q in range(b):
        result.append(
            [
                {
                    "ids": sequences[q, j, : sizes[q, j]].tolist(),
                    "log_probability": float(probs[q, j]),
                    "eos": bool(finished[q, j]),
                    "length_exhausted": not bool(finished[q, j]),
                }
                for j in range(k)
                if np.isfinite(probs[q, j])
            ]
        )
    return result
