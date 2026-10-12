import pytest
import torch
from torch_geometric.data import Batch

from structure_pretrain.config import GraphConfig
from structure_pretrain.data.molecular_graph import smiles_to_graph
from structure_pretrain.models.graph_encoder import GraphEncoder


def test_graph_forward_and_atom_order_invariance():
    encoder = GraphEncoder(GraphConfig()).eval()
    graphs = Batch.from_data_list([smiles_to_graph("CCO"), smiles_to_graph("OCC")])
    with torch.no_grad():
        encoded = encoder(graphs)
    assert encoded.shape == (2, 256)
    assert torch.isfinite(encoded).all()
    torch.testing.assert_close(encoded[0], encoded[1], atol=1e-5, rtol=1e-5)


def test_atom_ring_feature_and_bond_types():
    graph = smiles_to_graph("C1=CC=CC=C1C#CC=C")
    assert graph.x.shape[1] == 7
    assert graph.edge_attr.shape[1] == 3
    assert set(graph.edge_attr[:, 0].tolist()) == {0, 1, 2, 3}
    assert graph.x[:, 5].sum() == 6
    assert graph.rwse.shape == (graph.num_nodes, 8)
    assert graph.rwse.min() >= 0 and graph.rwse.max() <= 1


def test_stereochemistry_not_encoded():
    left, right = smiles_to_graph("C/C=C/C"), smiles_to_graph("C/C=C\\C")
    for field in ("x", "edge_index", "edge_attr", "rwse"):
        torch.testing.assert_close(left[field], right[field])


@pytest.mark.parametrize("text", ["", "invalid", "[*:1]OCC", "C.[Na+]", "C[NH3+]"])
def test_reject_invalid_full_template(text):
    with pytest.raises(ValueError):
        smiles_to_graph(text)
