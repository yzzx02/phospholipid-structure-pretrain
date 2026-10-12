from torch import nn


class ArchitectureHead(nn.Module):
    def __init__(self, dimension: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dimension, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, global_state):
        return self.mlp(global_state)
