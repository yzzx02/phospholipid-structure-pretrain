from torch import nn


class ChainHead(nn.Module):
    def __init__(self, vocabulary_size: int, dimension: int = 512):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.mlp = nn.Sequential(
            nn.Linear(dimension, 256), nn.GELU(), nn.Linear(256, 4 * vocabulary_size)
        )

    def forward(self, global_state):
        return self.mlp(global_state).reshape(-1, 4, self.vocabulary_size)
