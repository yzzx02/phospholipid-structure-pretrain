import json

import torch

from structure_pretrain.config import load_config
from structure_pretrain.data.chain_vocab import DEMO_CHAIN_TOKENS
from structure_pretrain.data.smiles_tokenizer import SmilesTokenizer
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel


def main():
    counts = {}
    for name in ("graph", "fp"):
        config = load_config(f"configs/structure_pretrain_{name}.yaml")
        with torch.device("meta"):
            model = StructurePretrainModel(config.model, len(DEMO_CHAIN_TOKENS), SmilesTokenizer())
        counts[config.model.encoder] = {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "modules": {
                module_name: sum(parameter.numel() for parameter in module.parameters())
                for module_name, module in model.named_children()
            },
        }
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
