from dataclasses import dataclass, field

from structure_pretrain.config import GraphConfig, ModelConfig


@dataclass
class PolarTrainConfig:
    seed: int = 42
    batch_size: int = 16
    gradient_accumulation: int = 1
    num_workers: int = 2
    cpu_threads: int = 4
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    max_steps: int = 5000
    checkpoint_every: int = 500
    precision: str = "fp32"


@dataclass
class PolarConfig:
    version: str = "v0.4"
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(encoder="fingerprint", max_polar_length=84, dropout=0.2)
    )
    train: PolarTrainConfig = field(default_factory=PolarTrainConfig)
    max_top_k: int = 3
    h_fraction: float = 0.8
    minimum_similarity: float = 0.2


def validate_config(config):
    if (
        config.version != "v0.4"
        or config.train.precision != "fp32"
        or config.max_top_k not in (3, 5, 10)
        or config.model.max_polar_length != 84
        or config.model.encoder not in ("fingerprint", "graph")
        or config.model.graph != GraphConfig()
        or config.model.decoder_layers != 2
        or config.model.set_layers != 2
        or not 0 <= config.h_fraction <= 1
        or not 0 < config.train.max_steps <= 5000
        or not 0 < config.train.batch_size * config.train.gradient_accumulation <= 32
    ):
        raise ValueError("Unsupported bounded v0.4 configuration")
    return config
