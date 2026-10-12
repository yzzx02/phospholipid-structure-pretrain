import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool

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
        if config.pooling != "sum" or config.hidden_dim % config.attention_heads:
            raise ValueError("Use sum pooling and a hidden width divisible by attention heads")
        dimension = config.hidden_dim
        self.atom_encoder = CategoricalFeatures(ATOM_CARDINALITIES, dimension)
        self.bond_encoder = CategoricalFeatures(BOND_CARDINALITIES, dimension)
        self.rwse_encoder = nn.Linear(config.rwse_steps, dimension)
        self.layers = nn.ModuleList()
        for _ in range(config.layers):
            local_mlp = nn.Sequential(
                nn.Linear(dimension, 2 * dimension),
                nn.GELU(),
                nn.Linear(2 * dimension, dimension),
            )
            self.layers.append(
                GPSConv(
                    channels=dimension,
                    conv=GINEConv(local_mlp, train_eps=True),
                    heads=config.attention_heads,
                    dropout=dropout,
                    norm="layer_norm",
                    attn_type="multihead",
                )
            )
        self.readout = nn.Sequential(
            nn.Linear(dimension, 256), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(256)
        )

    def forward(self, graphs: Batch):
        state = self.atom_encoder(graphs.x) + self.rwse_encoder(graphs.rwse)
        bonds = self.bond_encoder(graphs.edge_attr)
        for layer in self.layers:
            state = layer(state, graphs.edge_index, graphs.batch, edge_attr=bonds)
        return self.readout(global_add_pool(state, graphs.batch, size=graphs.num_graphs))
