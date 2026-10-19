"""Audit two samples with 3+2 real templates, including complete-model backward."""

import json

import torch

from structure_pretrain.config import ExperimentConfig
from structure_pretrain.data.batching import collate_structure_examples
from structure_pretrain.data.dummy_structure_dataset import DummyStructureDataset
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel
from structure_pretrain.training.losses import pretraining_loss
from structure_pretrain.utils.formula import FormulaStandardizer

TEMPLATE_FIELDS = (
    "template_ids",
    "template_smiles",
    "template_formulas",
    "delta_formulas",
    "template_mask",
    "structures",
)


def audit_encoder(encoder: str) -> dict:
    torch.manual_seed(42)
    config = ExperimentConfig()
    config.model.encoder = encoder
    dataset = DummyStructureDataset(encoder)
    examples = [
        {
            key: value[:count] if key in TEMPLATE_FIELDS else value
            for key, value in dataset[row].items()
        }
        for row, count in enumerate((3, 2))
    ]
    batch = collate_structure_examples(examples)
    model = StructurePretrainModel(
        config.model,
        len(dataset.chain_vocab),
        dataset.tokenizer,
        FormulaStandardizer.fit(dataset.examples[:6]),
    ).train()
    shapes, hooks, captured = {}, [], {}

    def shape_hook(name):
        def hook(module, arguments, output):
            shapes[name] = list(output.shape)

        return hook

    if encoder == "graph":
        graphs = batch["graphs"]
        shapes.update(
            {key: list(graphs[key].shape) for key in ("x", "edge_index", "edge_attr", "batch")}
        )
        for name, module in (
            ("node_h", model.structure_encoder.atom_encoder),
            ("edge_h", model.structure_encoder.bond_encoder),
        ):
            hooks.append(module.register_forward_hook(shape_hook(name)))
        for index, layer in enumerate(model.structure_encoder.layers):
            hooks.append(layer.register_forward_hook(shape_hook(f"gine_{index + 1}")))

        def pooled_hook(module, arguments):
            shapes["pooled"] = list(arguments[0].shape)

        hooks.append(model.structure_encoder.readout.register_forward_pre_hook(pooled_hook))
    else:
        shapes["fingerprints"] = list(batch["fingerprints"].shape)

    def encoded_hook(module, arguments, output):
        shapes["structure_embedding"] = list(output.shape)
        captured["flat"] = output.detach().clone()

    def padded_hook(module, arguments):
        shapes["padded_structure_embedding"] = list(arguments[0].shape)
        captured["padded"] = arguments[0].detach().clone()

    hooks.append(model.structure_encoder.register_forward_hook(encoded_hook))
    hooks.append(model.projector.register_forward_pre_hook(padded_hook))
    output = model(batch)
    losses = pretraining_loss(output, batch, dataset.tokenizer.pad_id, config.train)
    losses["loss"].backward()
    for hook in hooks:
        hook.remove()
    mask = batch["template_mask"]
    assert mask.sum(-1).tolist() == [3, 2]
    assert shapes["structure_embedding"] == [5, 256]
    assert shapes["padded_structure_embedding"] == [2, 10, 256]
    torch.testing.assert_close(captured["padded"][mask], captured["flat"])
    assert captured["padded"][~mask].count_nonzero() == 0
    if encoder == "graph":
        assert batch["graphs"].num_graphs == 5
        assert shapes["node_h"] == [shapes["x"][0], 128]
        assert shapes["edge_h"] == [shapes["edge_attr"][0], 128]
        assert shapes["pooled"] == [5, 128]
    unused = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not unused and not nonfinite
    assert all(torch.isfinite(value).all() for value in output.values())
    assert all(torch.isfinite(value) for value in losses.values())
    shapes.update(
        {
            key: list(output[key].shape)
            for key in (
                "template_tokens",
                "template_memory",
                "chain_logits",
                "polar_logits",
            )
        }
    )
    return {
        "encoder": encoder,
        "B": 2,
        "K": 10,
        "M": 5,
        "shapes": shapes,
        "template_mask": mask.int().tolist(),
        "padding_order": "passed",
        "losses": {name: value.detach().item() for name, value in losses.items()},
        "unused_parameters": unused,
        "nonfinite_gradients": nonfinite,
        "nonfinite_outputs": False,
        "forward_loss_backward": "passed",
    }


def main():
    torch.set_num_threads(2)
    print(
        json.dumps(
            {encoder: audit_encoder(encoder) for encoder in ("graph", "fingerprint")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
