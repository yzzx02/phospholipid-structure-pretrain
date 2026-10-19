"""Vocabulary supplied by future data builder; only a tiny test vocabulary is bundled."""

from structure_pretrain.library_builder.chain_library import chain_sort_key

DEMO_CHAIN_TOKENS = ("NULL", "A-16:0", "A-18:0", "A-18:1", "O-16:0", "P-18:0")


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
        if not 1 <= len(chains) <= 2:
            raise ValueError(
                "Targets must contain 1..2 chains in canonical slots, not sn positions"
            )
        return [self.index[token] for token in sorted(chains, key=chain_sort_key)] + [0] * (
            2 - len(chains)
        )
