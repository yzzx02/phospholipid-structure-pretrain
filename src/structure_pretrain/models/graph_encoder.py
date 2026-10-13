import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_add_pool

from structure_pretrain.config import GraphConfig
from structure_pretrain.data.molecular_graph import ATOM_CARDINALITIES, BOND_CARDINALITIES


class CategoricalFeatures(nn.Module):
    def __init__(self, cardinalities: tuple[int, ...], dimension: int):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(size, dimension) for size in cardinalities)

    def forward(self, features: torch.Tensor):
        return sum(embedding(features[:, i]) for i, embedding in enumerate(self.embeddings))


class GraphEncoder(nn.Module):
    def __init__(self, config: GraphConfig, dropout: float = 0.1):
        super().__init__()
        if config != GraphConfig():
            raise ValueError("Use fixed GINE / 3 layers / hidden 128 / sum / output 256")
        dimension = config.hidden_dim
        self.atom_encoder = CategoricalFeatures(ATOM_CARDINALITIES, dimension)
        self.bond_encoder = CategoricalFeatures(BOND_CARDINALITIES, dimension)
        self.layers = nn.ModuleList()
        for _ in range(config.layers):
            local_mlp = nn.Sequential(
                nn.Linear(dimension, dimension),
                nn.GELU(),
                nn.Linear(dimension, dimension),
            )
            self.layers.append(GINEConv(local_mlp, train_eps=True))
        self.norms = nn.ModuleList(nn.LayerNorm(dimension) for _ in self.layers)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(dimension, config.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(config.output_dim),
        )

    def forward(self, graphs: Batch):
        """Categorical atom/bond fields -> real graph embeddings [M,256], before padding."""
        if graphs.x.ndim != 2 or graphs.x.shape[1] != 7:
            raise ValueError("Expected atom categories [N_atoms,7]")
        if graphs.edge_index.ndim != 2 or graphs.edge_index.shape[0] != 2:
            raise ValueError("Expected directed edge_index [2,N_edges]")
        if graphs.edge_attr.shape != (graphs.edge_index.shape[1], 3):
            raise ValueError("Expected bond categories [N_edges,3]")
        if graphs.batch.shape != (graphs.x.shape[0],):
            raise ValueError("Expected graph membership batch [N_atoms]")
        state = self.atom_encoder(graphs.x)
        bonds = self.bond_encoder(graphs.edge_attr)
        for layer, norm in zip(self.layers, self.norms):
            state = self.dropout(norm(self.activation(layer(state, graphs.edge_index, bonds))))
        return self.readout(global_add_pool(state, graphs.batch, size=graphs.num_graphs))
