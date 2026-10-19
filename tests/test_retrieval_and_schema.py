from dataclasses import replace

import pytest
import torch

from structure_pretrain.config import RetrievalConfig
from structure_pretrain.data.chain_vocab import DEMO_CHAIN_TOKENS, ChainVocabulary
from structure_pretrain.data.dummy_structure_dataset import dummy_records
from structure_pretrain.data.molecular_graph import structure_key
from structure_pretrain.data.schema import StructureRecord
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.data.structure_dataset import StructurePretrainingDataset
from structure_pretrain.retrieval.structural_retrieval import MorganStructuralRetriever


def test_exclude_target_canonical_duplicates_and_deduplicate_bank():
    library, _ = dummy_records()
    aliases = [
        replace(library[0], structure_id="alias-self", smiles="OP(=O)(O)OCC"),
        replace(library[1], structure_id="alias-other"),
    ]
    retriever = MorganStructuralRetriever(library + aliases, RetrievalConfig())
    hits = retriever.retrieve(library[0])
    keys = [structure_key(hit.template.smiles) for hit in hits]
    assert keys and len(keys) == len(set(keys))
    assert structure_key(library[0].smiles) not in keys
    assert all(hit.template.structure_id != library[0].structure_id for hit in hits)
    assert all(0.4 <= hit.structure_similarity <= 1 for hit in hits)


def test_max_top10_and_zero_hits_reported():
    library, targets = dummy_records()
    library += [
        StructureRecord(
            f"extra-{n}", "C" * n + "OP(=O)(O)O", (n, 2 * n + 3, 0, 4, 1, 0, 0, 0, 0, 0)
        )
        for n in range(12, 24)
    ]
    retriever = MorganStructuralRetriever(library, RetrievalConfig(minimum_similarity=0))
    assert len(retriever.retrieve(library[0])) == 10
    zero = MorganStructuralRetriever([library[0]], RetrievalConfig())
    with pytest.raises(ValueError, match="Zero-template"):
        StructurePretrainingDataset(
            [targets[0]], zero, ChainVocabulary(DEMO_CHAIN_TOKENS), SmilesTokenizer(), "fingerprint"
        )


def test_supplied_formula_difference_and_chain_sort():
    vocab = ChainVocabulary(DEMO_CHAIN_TOKENS)
    assert vocab.encode(["A-18:1", "A-16:0"]) == [1, 3]
    assert vocab.encode(["A-18:1", "O-16:0"]) == [4, 3]
    assert vocab.encode(["A-18:1"]) == [3, 0]
    with pytest.raises(ValueError, match="1..2"):
        vocab.encode(["A-18:1", "A-16:0", "P-18:0"])
    from structure_pretrain.data.dummy_structure_dataset import DummyStructureDataset

    dataset = DummyStructureDataset("fingerprint")
    item = dataset[0]
    torch.testing.assert_close(
        item["delta_formulas"], item["target_formula"] - item["template_formulas"]
    )


def test_query_without_supplied_formula():
    library, targets = dummy_records()
    with pytest.raises(ValueError, match="mandatory"):
        replace(targets[0].structure, neutral_formula=None)
