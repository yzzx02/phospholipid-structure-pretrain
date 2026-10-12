"""SMILES lexical tokens; bracket atoms and mapped attachments stay intact."""

import re

import torch

TOKEN_PATTERN = re.compile(r"\[[^\]]+\]|Br|Cl|Si|Se|%\d{2}|[A-Za-z0-9*().=#+\-:/\\@]")
SPECIAL = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
BASE_TOKENS = (
    "B",
    "C",
    "N",
    "O",
    "P",
    "S",
    "F",
    "I",
    "Br",
    "Cl",
    "c",
    "n",
    "o",
    "s",
    "(",
    ")",
    "=",
    "#",
    "-",
    "+",
    ".",
    ":",
    "*",
    "/",
    "\\",
    "@",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "[*:1]",
    "[*:2]",
    "[N+]",
    "[O-]",
    "[NH3+]",
    "[NH2+]",
    "[nH]",
)


class SmilesTokenizer:
    def __init__(self, extra_tokens: list[str] | tuple[str, ...] = ()):
        self.tokens = tuple(dict.fromkeys(SPECIAL + BASE_TOKENS + tuple(extra_tokens)))
        self.index = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = range(4)

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, smiles: str) -> list[int]:
        pieces = TOKEN_PATTERN.findall(smiles)
        if "".join(pieces) != smiles:
            raise ValueError(f"Unsupported SMILES lexical syntax: {smiles}")
        unknown = [piece for piece in pieces if piece not in self.index]
        if unknown:
            raise ValueError(f"Extend the training vocabulary for tokens: {unknown}")
        return [self.bos_id] + [self.index[piece] for piece in pieces] + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        result = []
        for index in ids:
            if index == self.eos_id:
                break
            if index in (self.pad_id, self.bos_id):
                continue
            if index == self.unk_id:
                raise ValueError("Cannot decode an unknown SMILES token")
            result.append(self.tokens[index])
        return "".join(result)

    def pad(self, strings: list[str]) -> torch.Tensor:
        """Return BOS...EOS sequences [B,L], right-padded with PAD."""
        encoded = [self.encode(smiles) for smiles in strings]
        tensor = torch.full((len(encoded), max(map(len, encoded))), self.pad_id, dtype=torch.long)
        for row, ids in enumerate(encoded):
            tensor[row, : len(ids)] = torch.tensor(ids)
        return tensor
