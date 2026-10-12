"""Toy alkyl phosphates with artificial output labels; NOT a phospholipid training set."""

from structure_pretrain.config import RetrievalConfig
from structure_pretrain.data.chain_vocab import DEMO_CHAIN_TOKENS, ChainVocabulary
from structure_pretrain.data.schema import StructureRecord, TargetRecord
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.data.structure_dataset import StructurePretrainingDataset
from structure_pretrain.retrieval.structural_retrieval import MorganStructuralRetriever


def dummy_records():
    # These ten supplied formulas are fixtures, not formulas inferred with RDKit.
    formula_rows = [
        (2, 7),
        (3, 9),
        (4, 11),
        (5, 13),
        (6, 15),
        (7, 17),
        (8, 19),
        (9, 21),
        (10, 23),
        (11, 25),
    ]
    library = [
        StructureRecord(f"toy-{n:02d}", "C" * n + "OP(=O)(O)O", (n, h, 0, 4, 1, 0, 0, 0, 0, 0))
        for n, h in formula_rows
    ]
    # Cover chain counts 1/2/3/4 and missing formulas, without defining real cutting rules.
    artificial_chains = [
        ("A-16:0",),
        ("A-18:1", "A-16:0"),
        ("A-18:0", "O-16:0", "A-16:0"),
        ("A-18:1", "A-16:0", "P-18:0", "A-18:0"),
    ]
    heads = ("[*:1]OCCN", "[*:1]O", "[*:1]OCCO", "[*:1]OCC(O)CO")
    targets = [
        TargetRecord(record, int(i % 4 >= 2), artificial_chains[i % 4], heads[i % 4], i % 3 != 1)
        for i, record in enumerate(library[:8])
    ]
    return library, targets


class DummyStructureDataset(StructurePretrainingDataset):
    def __init__(self, encoder: str = "graph", retrieval: RetrievalConfig | None = None):
        self.chain_vocab = ChainVocabulary(DEMO_CHAIN_TOKENS)
        self.tokenizer = SmilesTokenizer()
        library, targets = dummy_records()
        retriever = MorganStructuralRetriever(library, retrieval or RetrievalConfig())
        super().__init__(targets, retriever, self.chain_vocab, self.tokenizer, encoder)
