import torch
from torch_geometric.nn import GINEConv

from structure_pretrain.config import ExperimentConfig
from structure_pretrain.data.batching import collate_structure_examples
from structure_pretrain.data.dummy_structure_dataset import DummyStructureDataset
from structure_pretrain.models.structure_pretrain_model import StructurePretrainModel


def test_five_real_graphs_then_template_padding():
    dataset = DummyStructureDataset("graph")
    fields = (
        "template_ids",
        "template_smiles",
        "template_formulas",
        "delta_formulas",
        "template_mask",
        "structures",
    )
    examples = [
        {key: value[:count] if key in fields else value for key, value in dataset[row].items()}
        for row, count in enumerate((3, 2))
    ]
    batch = collate_structure_examples(examples)
    model = StructurePretrainModel(
        ExperimentConfig().model, len(dataset.chain_vocab), dataset.tokenizer
    ).eval()
    encoder = model.structure_encoder
    assert len(encoder.layers) == 3 and all(isinstance(layer, GINEConv) for layer in encoder.layers)
    graph = batch["graphs"]
    assert graph.num_graphs == 5
    assert graph.x.shape[1] == 7 and graph.x.dtype == torch.long
    assert graph.edge_index.shape[0] == 2 and graph.edge_index.dtype == torch.long
    assert graph.edge_attr.shape == (graph.edge_index.shape[1], 3)
    assert graph.edge_attr.dtype == torch.long
    assert graph.batch.shape == (graph.x.shape[0],)
    assert encoder.atom_encoder(graph.x).shape == (graph.x.shape[0], 128)
    assert encoder.bond_encoder(graph.edge_attr).shape == (graph.edge_attr.shape[0], 128)
    assert batch["template_mask"].int().tolist() == [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    captured = {}

    def pooled_hook(module, inputs):
        captured["pooled"] = inputs[0].detach().clone()

    def flat_hook(module, inputs, output):
        captured["flat"] = output.detach().clone()

    def padded_hook(module, inputs):
        captured["padded"] = inputs[0].detach().clone()

    hooks = [
        encoder.readout.register_forward_pre_hook(pooled_hook),
        encoder.register_forward_hook(flat_hook),
        model.projector.register_forward_pre_hook(padded_hook),
    ]
    with torch.no_grad():
        output = model(batch)
    for hook in hooks:
        hook.remove()
    assert captured["pooled"].shape == (5, 128)
    assert captured["flat"].shape == (5, 256)
    assert captured["padded"].shape == (2, 10, 256)
    torch.testing.assert_close(captured["padded"][batch["template_mask"]], captured["flat"])
    assert captured["padded"][~batch["template_mask"]].count_nonzero() == 0
    assert output["template_memory"].shape == (2, 11, 512)
