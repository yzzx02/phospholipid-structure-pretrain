"""Interfaces only: real structures, labels, vocabularies and formulas arrive externally."""

from dataclasses import dataclass

from structure_pretrain.utils.formula import validate_formula


@dataclass(frozen=True)
class StructureRecord:
    structure_id: str
    smiles: str
    neutral_formula: tuple[int, ...]

    def __post_init__(self):
        if not self.structure_id or not self.smiles:
            raise ValueError("Structure ID and full SMILES are required")
        if self.neutral_formula is None:
            raise ValueError("Trusted neutral formula is mandatory")
        validate_formula(self.neutral_formula)


@dataclass(frozen=True)
class TargetRecord:
    structure: StructureRecord
    chain_tokens: tuple[str, ...]
    polar_fragment_smiles: str

    def __post_init__(self):
        if not 1 <= len(self.chain_tokens) <= 2 or not self.polar_fragment_smiles:
            raise ValueError("Target labels must be supplied externally")
