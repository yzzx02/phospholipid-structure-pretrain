"""Conservative two-port SMILES grammar and exact online heavy-atom budgets.

H suffixes, charges, ring digits and atom-map digits are not element counts.
Implicit/explicit H and the neutral H-capped formula are checked finally in RDKit.
"""

from dataclasses import dataclass, replace

import numpy as np

from structure_pretrain.polar_only_v04.constrained_decode import BONDS, RING, token_element
from structure_pretrain.utils.formula import ELEMENTS

HEAVY = tuple(e for e in ELEMENTS if e != "H")
PORTS = ("[*:101]", "[*:102]")


@dataclass(frozen=True)
class State:
    atoms: tuple = ()
    current: int = -1
    branches: tuple = ()
    rings: tuple = ()
    edges: frozenset = frozenset()
    pending: str = ""
    last: str = "start"
    ports: int = 0

    @property
    def complete(self):
        return (
            bool(self.atoms)
            and self.ports == 3
            and not self.branches
            and not self.rings
            and not self.pending
            and self.last in ("atom", "ring", "close")
        )

    @property
    def on_port(self):
        return self.current >= 0 and self.atoms[self.current] == "*"


def step(state, token):
    """Pure syntax transition; None means forbidden, never repaired."""
    if token == "(":
        if state.current < 0 or state.on_port or state.last not in ("atom", "ring", "close"):
            return None
        return replace(
            state, branches=state.branches + ((state.current, len(state.atoms)),), last="open"
        )
    if token == ")":
        if not state.branches or state.pending or state.last not in ("atom", "ring", "close"):
            return None
        parent, first = state.branches[-1]
        if len(state.atoms) <= first:
            return None
        return replace(state, current=parent, branches=state.branches[:-1], last="close")
    if token in BONDS:
        if (
            state.current < 0
            or state.on_port
            or state.pending
            or state.last not in ("atom", "ring", "close", "open")
        ):
            return None
        return replace(state, pending=token, last="bond")
    if RING.fullmatch(token):
        if (
            state.current < 0
            or state.on_port
            or state.last not in ("atom", "ring", "close", "bond")
        ):
            return None
        if state.branches and len(state.atoms) <= state.branches[-1][1]:
            return None
        opened = {label: (atom, bond) for label, atom, bond in state.rings}
        edges = state.edges
        if token in opened:
            other, bond = opened.pop(token)
            edge = tuple(sorted((other, state.current)))
            if (
                other == state.current
                or edge in edges
                or (bond and state.pending and bond != state.pending)
            ):
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
    if symbol is None or state.on_port:
        return None
    ports = state.ports
    if symbol == "*":
        if token not in PORTS or state.current < 0 or state.atoms[state.current] != "O":
            return None
        bit = 1 << PORTS.index(token)
        if ports & bit or state.pending not in ("", "-"):
            return None
        ports |= bit
    elif symbol not in HEAVY:
        return None
    index = len(state.atoms)
    edges = state.edges | ({tuple(sorted((state.current, index)))} if state.current >= 0 else set())
    return replace(
        state,
        atoms=state.atoms + (symbol,),
        current=index,
        edges=frozenset(edges),
        ports=ports,
        pending="",
        last="atom",
    )


class Budget:
    def __init__(self, tokenizer, formulas):
        formulas = np.asarray(formulas)
        if formulas.ndim != 2 or formulas.shape[1] != 10 or not np.isfinite(formulas).all():
            raise ValueError("Supply neutral whole-core formula[B,10]")
        if (formulas < 0).any() or not np.array_equal(formulas, formulas.round()):
            raise ValueError("Nonnegative integer atom counts required")
        self.target = np.column_stack(
            (
                formulas[:, [ELEMENTS.index(e) for e in HEAVY]],
                np.ones((len(formulas), 2), dtype=np.int64),
            )
        ).astype(np.int16)
        self.tokenizer = tokenizer
        self.increments = np.zeros((len(tokenizer), len(HEAVY) + 2), dtype=np.int16)
        self.countable = np.zeros(len(tokenizer), dtype=bool)
        self.syntax_ids = []
        self.atom_ids = []
        self.port_ids = [tokenizer.index[x] for x in PORTS]
        self.bond_ids = [tokenizer.index[x] for x in BONDS if x in tokenizer.index]
        self.ring_ids = {
            token: i for i, token in enumerate(tokenizer.tokens) if RING.fullmatch(token)
        }
        self.open_id, self.close_id = tokenizer.index["("], tokenizer.index[")"]
        for i, token in enumerate(tokenizer.tokens):
            if i in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id):
                continue
            try:
                symbol = token_element(token)
            except ValueError:
                continue
            if symbol in HEAVY:
                self.increments[i, HEAVY.index(symbol)] = 1
                self.countable[i] = True
                self.atom_ids.append(i)
            elif token in PORTS:
                self.increments[i, len(HEAVY) + PORTS.index(token)] = 1
                self.countable[i] = True
            elif symbol is None and (
                token in BONDS or token in ("(", ")") or RING.fullmatch(token)
            ):
                self.countable[i] = True
                self.syntax_ids.append(i)
        self.countable[tokenizer.eos_id] = True

    def allowed(self, consumed, queries, states=None):
        targets = self.target[np.asarray(queries)]
        mask = self.countable[None] & np.all(
            consumed[:, None, :] + self.increments[None] <= targets[:, None, :], axis=-1
        )
        mask[:, self.tokenizer.eos_id] = np.all(consumed == targets, axis=1)
        if states is not None:
            for row, state in enumerate(states):
                if state.on_port:
                    mask[row, self.atom_ids] = False
                for i in self.port_ids:
                    token = self.tokenizer.tokens[i]
                    bit = 1 << PORTS.index(token)
                    mask[row, i] &= (
                        state.current >= 0
                        and not state.on_port
                        and state.atoms[state.current] == "O"
                        and not (state.ports & bit)
                        and state.pending in ("", "-")
                    )
                mask[row, self.open_id] &= (
                    state.current >= 0
                    and not state.on_port
                    and state.last in ("atom", "ring", "close")
                )
                mask[row, self.close_id] &= (
                    bool(state.branches)
                    and not state.pending
                    and state.last in ("atom", "ring", "close")
                    and len(state.atoms) > state.branches[-1][1]
                )
                mask[row, self.bond_ids] &= (
                    state.current >= 0
                    and not state.on_port
                    and not state.pending
                    and state.last in ("atom", "ring", "close", "open")
                )
                ring_context = (
                    state.current >= 0
                    and not state.on_port
                    and state.last in ("atom", "ring", "close", "bond")
                    and (not state.branches or len(state.atoms) > state.branches[-1][1])
                )
                if not ring_context:
                    mask[row, list(self.ring_ids.values())] = False
                else:
                    for label, other, bond in state.rings:
                        i = self.ring_ids.get(label)
                        if i is not None:
                            edge = tuple(sorted((other, state.current)))
                            mask[row, i] &= (
                                other != state.current
                                and edge not in state.edges
                                and not (bond and state.pending and bond != state.pending)
                            )
                mask[row, self.tokenizer.eos_id] &= state.complete
        return mask
