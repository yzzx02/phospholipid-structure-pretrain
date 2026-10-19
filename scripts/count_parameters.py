import argparse
import json
from pathlib import Path

import torch

from structure_pretrain.config import load_config
from structure_pretrain.data.chain_vocab import DEMO_CHAIN_TOKENS
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel


def main():
    parser = argparse.ArgumentParser(
        description="Count actual prepared v0.3 or explicit demo shapes"
    )
    parser.add_argument("--prepared", default="data/private/pretraining_v03")
    args = parser.parse_args()
    prepared = Path(args.prepared)
    chain_count, tokenizer, length = len(DEMO_CHAIN_TOKENS), SmilesTokenizer(), 64
    scope = "demo_vocabulary_not_real_v03"
    if (prepared / "polar_tokenizer.json").exists():
        vocab = json.loads((prepared / "polar_tokenizer.json").read_text(encoding="utf-8"))
        chain_count = len(
            json.loads((prepared / "chain_vocab.json").read_text(encoding="utf-8"))["tokens"]
        )
        tokenizer = SmilesTokenizer(vocab["tokens"])
        length, scope = vocab["max_polar_length"], "prepared_v0.3_real_vocabulary"
    counts = {}
    for name in ("graph", "fp"):
        config = load_config(f"configs/structure_pretrain_{name}.yaml")
        config.model.max_polar_length = length
        with torch.device("meta"):
            model = StructurePretrainModel(config.model, chain_count, tokenizer)
        counts[config.model.encoder] = {
            "scope": scope,
            "chain_vocabulary": chain_count,
            "polar_vocabulary": len(tokenizer),
            "max_polar_length": length,
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "modules": {
                module_name: sum(parameter.numel() for parameter in module.parameters())
                for module_name, module in model.named_children()
            },
        }
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
