"""Inference-only element budgets and conservative SMILES prefix constraints.

This module does not change decoder parameters, tokenization or training labels.
Hydrogen is deliberately checked only by the final RDKit/Builder validator.
"""

import math
import re
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import torch

from structure_pretrain.utils.formula import ELEMENTS

HEAVY = tuple(e for e in ELEMENTS if e != "H")
BRACKET_ATOM = re.compile(r"^\[(?:\d+)?(\*|[A-Z][a-z]?|se|as|[bcnops])")
RING = re.compile(r"^(?:\d|%\d{2})$")
BONDS = frozenset(("-", "=", "#", ":", "/", "\\"))


def token_element(token):
    """Return one atom symbol, '*' for a dummy, None for syntax; fail closed.

    Bracket H suffixes, isotope numbers and atom-map numbers are not atoms.
    An explicit [H] is an atom, but remains outside this heavy-element budget.
    """
    if token.startswith("["):
        match = BRACKET_ATOM.match(token)
        if not match or not token.endswith("]"):
            raise ValueError(f"Uncountable bracket token: {token}")
        return match[1].capitalize() if match[1] != "*" else "*"
    if token == "*":
        return "*"
    if re.fullmatch(r"[A-Z][a-z]?|se|as|[bcnops]", token):
        return token.capitalize()
    if token in BONDS or token in ("(", ")", ".", "+", "@") or RING.fullmatch(token):
        return None
    raise ValueError(f"Uncountable lexical token: {token}")


@dataclass(frozen=True)
class SyntaxState:
    """Single-component notation, balanced branches/rings, mapped P boundary.

    This is a conservative lexical grammar, not a valence/aromaticity solver.
    Ring labels may be reused after closure; multiple phosphorus atoms are legal.
    The attachment is located by [*:1], never by the first P in a SMILES string.
    """

    atoms: tuple = ()
    current: int = -1
    branches: tuple = ()
    rings: tuple = ()
    edges: frozenset = frozenset()
    pending: str = ""
    last: str = "start"
    dummy: int = -1
    dummy_degree: int = 0

    @property
    def complete(self):
        return bool(
            self.atoms
            and not self.branches
            and not self.rings
            and not self.pending
            and self.last in ("atom", "ring", "close")
            and self.dummy >= 0
            and self.dummy_degree == 1
        )


@lru_cache(maxsize=65536)
def syntax_step(state, token):
    """Return a valid prefix state, or None; never repair/complete a prefix."""
    if token == "(":
        if state.last not in ("atom", "ring", "close") or state.current < 0:
            return None
        if state.current == state.dummy and state.dummy_degree:
            return None
        return replace(
            state, branches=state.branches + ((state.current, len(state.atoms)),), last="open"
        )
    if token == ")":
        if not state.branches or state.last not in ("atom", "ring", "close"):
            return None
        parent, first = state.branches[-1]
        if len(state.atoms) <= first or state.pending:
            return None
        return replace(state, current=parent, branches=state.branches[:-1], last="close")
    if token in BONDS:
        if state.current < 0 or state.last not in ("atom", "ring", "close", "open"):
            return None
        if state.current == state.dummy and (state.dummy_degree or token != "-"):
            return None
        return replace(state, pending=token, last="bond")
    if RING.fullmatch(token):
        if state.current < 0 or state.last not in ("atom", "ring", "close", "bond"):
            return None
        if state.branches and len(state.atoms) <= state.branches[-1][1]:
            return None
        if state.current == state.dummy:
            return None
        opened = {label: (atom, bond) for label, atom, bond in state.rings}
        edges = state.edges
        if token in opened:
            other, bond = opened.pop(token)
            edge = tuple(sorted((other, state.current)))
            if other == state.current or edge in edges:
                return None
            if bond and state.pending and bond != state.pending:
                return None
            edges = edges | {edge}
        else:
            opened[token] = (state.current, state.pending)
        return replace(
            state,
            rings=tuple(sorted((k, *v) for k, v in opened.items())),
            edges=edges,
            pending="",
            last="ring",
        )
    try:
        symbol = token_element(token)
    except ValueError:
        return None
    if symbol is None or token in (".", "+", "@"):
        return None
    if symbol == "*" and (token != "[*:1]" or state.dummy >= 0):
        return None
    index = len(state.atoms)
    degree = state.dummy_degree
    dummy = index if symbol == "*" else state.dummy
    edges = state.edges
    if state.current >= 0:
        previous = state.atoms[state.current]
        if previous == "*" or symbol == "*":
            if state.pending not in ("", "-"):
                return None
            if (previous == "*" and (symbol != "P" or degree)) or (
                symbol == "*" and previous != "P"
            ):
                return None
            degree += 1
        edges = edges | {tuple(sorted((state.current, index)))}
    return replace(
        state,
        atoms=state.atoms + (symbol,),
        current=index,
        edges=edges,
        dummy=dummy,
        dummy_degree=degree,
        pending="",
        last="atom",
    )


class ElementBudget:
    def __init__(self, tokenizer, formula):
        formula = np.asarray(formula, dtype=np.float64)
        if (
            formula.shape != (10,)
            or not np.isfinite(formula).all()
            or (formula < 0).any()
            or not np.equal(formula, formula.round()).all()
            or formula[ELEMENTS.index("P")] < 1
        ):
            raise ValueError("A nonnegative integral P-containing Polar formula is required")
        self.target = formula[[ELEMENTS.index(e) for e in HEAVY]].astype(np.int64)
        self.tokenizer = tokenizer
        self.increments = np.zeros((len(tokenizer), len(HEAVY)), dtype=np.int64)
        self.countable = np.ones(len(tokenizer), dtype=bool)
        for i, token in enumerate(tokenizer.tokens):
            if i in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id):
                continue
            try:
                element = token_element(token)
            except ValueError:
                self.countable[i] = False
                continue
            if element in HEAVY:
                self.increments[i, HEAVY.index(element)] = 1
            elif element not in (None, "*", "H"):
                # E.g. B, Si, Se: absent from the trusted ten-element formula schema.
                self.countable[i] = False
        self.countable[[tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]] = False

    def allowed(self, consumed, syntax=None):
        mask = self.countable & np.all(consumed + self.increments <= self.target, axis=1)
        mask[self.tokenizer.eos_id] = np.array_equal(consumed, self.target)
        if syntax is not None:
            for i in np.flatnonzero(mask):
                if i == self.tokenizer.eos_id:
                    mask[i] = syntax.complete
                else:
                    mask[i] = syntax_step(syntax, self.tokenizer.tokens[i]) is not None
        return mask


@torch.no_grad()
def constrained_beam_search(
    decoder,
    memory,
    padding,
    tokenizer,
    polar_formula,
    *,
    beam_size=5,
    max_length=84,
    grammar=False,
    progress=None,
):
    """One query, existing weights and mean-log-probability beam retention.

    Probabilities use the *original* decoder distribution (with only its existing
    PAD/BOS/UNK mask). Constraint masking happens after log_softmax: feasible
    choices are not renormalized. EOS is never forced. Dead/unfinished beams
    remain explicit failures and are never repaired into a candidate.
    """
    if memory.shape[0] != 1 or padding.shape[0] != 1:
        raise ValueError("Supply exactly one independent query memory")
    if not 1 <= beam_size <= 128 or not 2 <= max_length <= decoder.max_length:
        raise ValueError("Bounded beam/length required")
    budget = ElementBudget(tokenizer, polar_formula)
    # ids, original log probability, finished, heavy counts, grammar state
    beams = [
        (
            [tokenizer.bos_id],
            0.0,
            False,
            np.zeros(len(HEAVY), dtype=np.int64),
            SyntaxState() if grammar else None,
        )
    ]
    for step in range(max_length - 1):
        active = [b for b in beams if not b[2]]
        if not active:
            break
        ids = torch.tensor([b[0] for b in active], device=memory.device)
        logits = decoder(ids, memory.expand(len(active), -1, -1), padding.expand(len(active), -1))[
            :, -1
        ].clone()
        logits[:, [tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]] = -torch.inf
        logp = logits.log_softmax(-1)
        if torch.isnan(logp).any() or torch.isposinf(logp).any():
            raise ValueError("Nonfinite decoder probability distribution")
        masks = np.stack([budget.allowed(b[3], b[4]) for b in active])
        logp.masked_fill_(~torch.as_tensor(masks, device=memory.device), -torch.inf)
        values, choices = logp.topk(min(beam_size, logp.shape[-1] - 3))
        candidates = [b for b in beams if b[2]]
        for i, (prefix, score, _, consumed, syntax) in enumerate(active):
            for value, token in zip(values[i].tolist(), choices[i].tolist()):
                if not math.isfinite(value):
                    continue
                done = token == tokenizer.eos_id
                next_syntax = (
                    syntax
                    if done or syntax is None
                    else syntax_step(syntax, tokenizer.tokens[token])
                )
                candidates.append(
                    (
                        prefix + [token],
                        score + value,
                        done,
                        consumed + budget.increments[token],
                        next_syntax,
                    )
                )
        beams = sorted(candidates, key=lambda b: b[1] / (len(b[0]) - 1), reverse=True)[:beam_size]
        if progress is not None:
            progress(step + 1)
    return [
        {"ids": ids, "log_probability": score, "eos": done, "length_exhausted": not done}
        for ids, score, done, _, _ in beams
    ]


def rank_unique_accepted(candidates):
    """No target-structure ranking and no database whitelist; graph dedup only."""
    ordered = sorted(candidates, key=lambda c: c["decoder_score"], reverse=True)
    selected, seen = [], set()
    for candidate in ordered:
        if not candidate["accepted"] or candidate["canonical_polar"] in seen:
            continue
        seen.add(candidate["canonical_polar"])
        selected.append(candidate)
    return selected
