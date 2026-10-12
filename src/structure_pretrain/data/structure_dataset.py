"""Dataset adapter over externally supplied records; no database building or label cutting."""

import torch
from torch.utils.data import Dataset

from structure_pretrain.data.chain_vocab import ChainVocabulary
from structure_pretrain.data.molecular_graph import fingerprint_tensor, smiles_to_graph
from structure_pretrain.data.schema import TargetRecord
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.retrieval.structural_retrieval import StructuralRetrieval


class StructurePretrainingDataset(Dataset):
    def __init__(
        self,
        targets: list[TargetRecord],
        retriever: StructuralRetrieval,
        chain_vocab: ChainVocabulary,
        tokenizer: SmilesTokenizer,
        encoder: str = "graph",
        rwse_steps: int = 8,
    ):
        if encoder not in {"graph", "fingerprint"}:
            raise ValueError("Unknown structure encoder")
        self.examples, self.skipped_target_ids = [], []
        cache = {}
        for target in targets:
            hits = retriever.retrieve(target.structure)
            if not hits:
                self.skipped_target_ids.append(target.structure.structure_id)
                continue
            formulas = torch.tensor([hit.template.neutral_formula for hit in hits]).float()
            available = torch.full((len(hits),), target.formula_available, dtype=torch.bool)
            query = (
                torch.tensor(target.structure.neutral_formula).float()
                if target.formula_available
                else torch.zeros(10)
            )
            delta = (
                query[None] - formulas if target.formula_available else torch.zeros_like(formulas)
            )
            smiles = [hit.template.smiles for hit in hits]
            for text in smiles:
                if text not in cache:
                    cache[text] = (
                        smiles_to_graph(text, rwse_steps)
                        if encoder == "graph"
                        else fingerprint_tensor(text)
                    )
            self.examples.append(
                {
                    "target_id": target.structure.structure_id,
                    "target_architecture": target.architecture,
                    "target_chain_tokens": chain_vocab.encode(target.chain_tokens),
                    "target_headgroup_token_ids": tokenizer.encode(target.headgroup_smiles),
                    # Do not expose the hidden query formula when it is unavailable.
                    "target_formula": query if target.formula_available else torch.zeros(10),
                    "template_ids": [hit.template.structure_id for hit in hits],
                    "template_smiles": smiles,
                    "template_formulas": formulas,
                    "delta_formulas": delta,
                    "structure_similarity": torch.tensor(
                        [hit.structure_similarity for hit in hits]
                    ),
                    "formula_available": available,
                    "template_mask": torch.ones(len(hits), dtype=torch.bool),
                    "structures": [cache[text] for text in smiles],
                    "encoder": encoder,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
