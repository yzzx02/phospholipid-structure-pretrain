import torch
from torch_geometric.data import Batch


def collate_structure_examples(examples: list[dict], max_top_k: int = 10, pad_id: int = 0) -> dict:
    if not examples or len({item["encoder"] for item in examples}) != 1:
        raise ValueError("Collate requires a nonempty batch with one structure encoder")
    batch_size = len(examples)
    mask = torch.zeros(batch_size, max_top_k, dtype=torch.bool)
    template_formula = torch.zeros(batch_size, max_top_k, 10)
    delta = torch.zeros_like(template_formula)
    available = torch.zeros_like(mask)
    similarity = torch.zeros(batch_size, max_top_k)
    length = max(len(item["target_headgroup_token_ids"]) for item in examples)
    tokens = torch.full((batch_size, length), pad_id, dtype=torch.long)
    structures = []
    for row, item in enumerate(examples):
        count = len(item["template_ids"])
        if not 1 <= count <= max_top_k or not item["template_mask"].all():
            raise ValueError("Dataset items must contain 1..K valid templates before padding")
        mask[row, :count] = True
        template_formula[row, :count] = item["template_formulas"]
        delta[row, :count] = item["delta_formulas"]
        available[row, :count] = item["formula_available"]
        similarity[row, :count] = item["structure_similarity"]
        ids = item["target_headgroup_token_ids"]
        tokens[row, : len(ids)] = torch.tensor(ids)
        structures.extend(item["structures"])
    batch = {
        "target_id": [item["target_id"] for item in examples],
        "target_architecture": torch.tensor([item["target_architecture"] for item in examples]),
        "target_chain_tokens": torch.tensor([item["target_chain_tokens"] for item in examples]),
        "target_headgroup_token_ids": tokens,
        "target_formula": torch.stack([item["target_formula"] for item in examples]),
        "template_ids": [item["template_ids"] for item in examples],
        "template_smiles": [item["template_smiles"] for item in examples],
        "template_formulas": template_formula,
        "delta_formulas": delta,
        "structure_similarity": similarity,
        "formula_available": available,
        "template_mask": mask,
    }
    # Flatten only real templates in mask row-major order; never graph-encode padded SMILES.
    if examples[0]["encoder"] == "graph":
        batch["graphs"] = Batch.from_data_list(structures)
    else:
        batch["fingerprints"] = torch.stack(structures)
    return batch
