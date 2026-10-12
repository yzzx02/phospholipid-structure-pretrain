"""Vocabulary supplied by future data builder; only a tiny test vocabulary is bundled."""

import re

PATTERN = re.compile(r"([AOP])-(\d+):(\d+)")
DEMO_CHAIN_TOKENS = ("NULL", "A-16:0", "A-18:0", "A-18:1", "O-16:0", "P-18:0")


def chain_sort_key(token: str):
    match = PATTERN.fullmatch(token)
    if not match:
        raise ValueError(f"Unsupported chain token: {token}")
    linkage, carbon, db = match.groups()
    # Deterministic representative ordering only; these slots have no sn meaning.
    return ({"O": 0, "P": 1, "A": 2}[linkage], int(carbon), int(db))


class ChainVocabulary:
    def __init__(self, tokens: list[str] | tuple[str, ...]):
        if not tokens or tokens[0] != "NULL" or len(set(tokens)) != len(tokens):
            raise ValueError("Vocabulary must start with NULL and contain unique tokens")
        for token in tokens[1:]:
            chain_sort_key(token)
        self.tokens = tuple(tokens)
        self.index = {token: index for index, token in enumerate(tokens)}

    def __len__(self):
        return len(self.tokens)

    def encode(self, chains: list[str] | tuple[str, ...]) -> list[int]:
        chains = [token for token in chains if token != "NULL"]
        if not 1 <= len(chains) <= 4:
            raise ValueError("Targets must contain 1..4 chains")
        return [self.index[token] for token in sorted(chains, key=chain_sort_key)] + [0] * (
            4 - len(chains)
        )
