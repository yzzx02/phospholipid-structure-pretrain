"""Interfaces only: real structures, labels, vocabularies and formulas arrive externally."""

from dataclasses import dataclass

from structure_pretrain.utils.formula import validate_formula


@dataclass(frozen=True)
class StructureRecord:
    structure_id: str
    smiles: str
    neutral_formula: tuple[int, ...] | None

    def __post_init__(self):
        if not self.structure_id or not self.smiles:
            raise ValueError("Structure ID and full SMILES are required")
        if self.neutral_formula is not None:
            validate_formula(self.neutral_formula)


@dataclass(frozen=True)
class TargetRecord:
    structure: StructureRecord
    architecture: int
    chain_tokens: tuple[str, ...]
    headgroup_smiles: str
    formula_available: bool = True

    def __post_init__(self):
        if self.architecture not in (0, 1):
            raise ValueError("Architecture must be mono=0 or bis=1")
        if not 1 <= len(self.chain_tokens) <= 4 or not self.headgroup_smiles:
            raise ValueError("Target labels must be supplied externally")
        if self.formula_available and self.structure.neutral_formula is None:
            raise ValueError("Available query formula must be supplied")
