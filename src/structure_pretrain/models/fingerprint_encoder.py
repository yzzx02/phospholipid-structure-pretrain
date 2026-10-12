from torch import nn


class FingerprintEncoder(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
        )

    def forward(self, fingerprints):
        return self.mlp(fingerprints)
