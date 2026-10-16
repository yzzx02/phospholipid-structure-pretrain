from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ELEMENTS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
HARD_MAX_STRUCTURES = 1_500_000
HARD_MAX_STRUCTURES_V02 = 2_000_000


class LibraryValidationError(ValueError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class Rejection:
    stage: str
    record_id: str
    reason: str
    message: str


@dataclass(frozen=True)
class ChainRule:
    chain_token: str
    linkage_type: str
    carbon_count: int
    double_bond_count: int
    representative_fragment_smiles: str
    representative_double_bond_positions: tuple[int, ...]
    source: str
    p_db_count_includes_vinyl: bool | None = None

    def row(self):
        row = asdict(self)
        row["representative_double_bond_positions"] = list(
            self.representative_double_bond_positions
        )
        return row


@dataclass(frozen=True)
class PolarRule:
    polar_id: str
    source: str
    source_id: str
    original_smiles: str
    polar_fragment_smiles: str
    attachment_atom_map: int
    polar_family: str
    allow_aa: bool
    allow_oa: bool
    allow_pa: bool
    source_polar_ids: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    def row(self):
        row = asdict(self)
        row["source_polar_ids"] = list(self.source_polar_ids)
        row["provenance"] = list(self.provenance)
        return row


@dataclass(frozen=True)
class ChainCombination:
    combination_id: str
    chain_tokens: tuple[str, ...]
    linkage_family: str
    total_carbon: int
    total_double_bonds: int
    is_lyso: bool
    mapped_from_p_tokens: tuple[str, ...] = ()

    def row(self):
        row = asdict(self)
        row["chain_tokens"] = list(self.chain_tokens)
        row["mapped_from_p_tokens"] = list(self.mapped_from_p_tokens)
        return row


@dataclass
class ChainSamplingConfig:
    short_max_total_c: int = 32
    medium_max_total_c: int = 40
    long_min_total_c: int = 41
    low_max_total_db: int = 2
    medium_max_total_db: int = 4


@dataclass
class BuildConfig:
    schema_version: str = "structure-library-v0.1"
    max_structures: int = 1_500_000
    max_extra_per_head: int = 100
    max_chain_combinations: int = 2_000_000
    seed: int = 42
    lyso_fraction_target: float = 0.10
    enable_pe_p: bool = True
    include_p_lyso: bool = False
    include_bis_phosphatidyl: bool = False
    bis_phosphatidyl_budget: int = 0
    parquet_batch_size: int = 2000
    anchor_combinations: list[str] = field(
        default_factory=lambda: [
            "A-16:0_A-18:1",
            "A-16:0_A-20:4",
            "A-18:0_A-18:1",
            "A-18:0_A-20:4",
            "A-18:0_A-22:6",
            "A-18:1_A-20:4",
            "A-18:1_A-18:1",
        ]
    )
    anchor_single_chains: list[str] = field(
        default_factory=lambda: [
            "A-16:0",
            "A-18:0",
            "A-18:1",
            "A-20:4",
            "A-22:6",
        ]
    )
    chain_sampling: ChainSamplingConfig = field(default_factory=ChainSamplingConfig)

    def validate(self):
        maximum = (
            HARD_MAX_STRUCTURES_V02
            if self.schema_version == "structure-library-v0.2"
            else HARD_MAX_STRUCTURES
        )
        if not 1 <= self.max_structures <= maximum:
            raise ValueError(f"max_structures must be between 1 and {maximum:,}")
        if self.max_extra_per_head < 0 or self.max_chain_combinations < 1:
            raise ValueError("Invalid per-head or combination-bank budget")
        if not 0 <= self.lyso_fraction_target < 1 or self.parquet_batch_size < 1:
            raise ValueError("Invalid lyso target or parquet batch size")
        if self.include_bis_phosphatidyl or self.bis_phosphatidyl_budget:
            raise NotImplementedError(
                "Bis-phosphatidyl requires a separate future rule implementation"
            )
        sampling = self.chain_sampling
        if not 0 < sampling.short_max_total_c < sampling.medium_max_total_c:
            raise ValueError("Carbon bins must increase")
        if sampling.long_min_total_c != sampling.medium_max_total_c + 1:
            raise ValueError("Carbon bins must be contiguous")
        if not 0 <= sampling.low_max_total_db < sampling.medium_max_total_db:
            raise ValueError("Double-bond bins must increase")


def load_build_config(path: str | Path) -> BuildConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
    raw.pop("inputs", None)
    raw.pop("output", None)
    sampling = ChainSamplingConfig(**raw.pop("chain_sampling", {}))
    config = BuildConfig(**raw, chain_sampling=sampling)
    config.validate()
    return config
