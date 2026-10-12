"""Independent retrieval interface; only structural Tanimoto similarity in v0.1."""

from dataclasses import dataclass
from typing import Protocol

from rdkit import DataStructs

from structure_pretrain.config import RetrievalConfig
from structure_pretrain.data.molecular_graph import fingerprint, structure_key
from structure_pretrain.data.schema import StructureRecord


@dataclass(frozen=True)
class TemplateHit:
    template: StructureRecord
    structure_similarity: float


class StructuralRetrieval(Protocol):
    def retrieve(self, target: StructureRecord) -> list[TemplateHit]: ...


class MorganStructuralRetriever:
    def __init__(self, library: list[StructureRecord], config: RetrievalConfig):
        if config.radius != 2 or config.fp_size != 2048:
            raise ValueError("v0.1 fixes radius=2 and fp_size=2048")
        if not 1 <= config.max_top_k <= 10 or not 0 <= config.minimum_similarity <= 1:
            raise ValueError("Retrieval requires K=1..10 and similarity in [0,1]")
        if len({record.structure_id for record in library}) != len(library):
            raise ValueError("Library IDs must be unique")
        if any(record.neutral_formula is None for record in library):
            raise ValueError("Every template must have a supplied neutral formula")
        # Duplicate identity ignores stereochemistry, matching this stage's feature granularity.
        # Retain first sorted ID deterministically; never rank repeated structures multiple times.
        unique = {}
        for record in sorted(library, key=lambda record: record.structure_id):
            unique.setdefault(structure_key(record.smiles), record)
        self.keys = list(unique)
        self.library = list(unique.values())
        self.fingerprints = [fingerprint(record.smiles) for record in self.library]
        self.config = config

    def retrieve(self, target: StructureRecord) -> list[TemplateHit]:
        key = structure_key(target.smiles)
        scores = DataStructs.BulkTanimotoSimilarity(fingerprint(target.smiles), self.fingerprints)
        hits = [
            TemplateHit(record, float(score))
            for record, candidate_key, score in zip(self.library, self.keys, scores)
            if record.structure_id != target.structure_id
            and candidate_key != key
            and score >= self.config.minimum_similarity
        ]
        return sorted(hits, key=lambda hit: (-hit.structure_similarity, hit.template.structure_id))[
            : self.config.max_top_k
        ]
