from dataclasses import dataclass, field

from omegaconf import OmegaConf


@dataclass
class GraphConfig:
    hidden_dim: int = 128
    layers: int = 3
    attention_heads: int = 4
    rwse_steps: int = 8
    pooling: str = "sum"


@dataclass
class ModelConfig:
    encoder: str = "graph"
    structure_dim: int = 256
    d_model: int = 512
    dropout: float = 0.1
    template_dropout: float = 0.2
    set_layers: int = 2
    set_heads: int = 8
    set_ffn_dim: int = 2048
    decoder_layers: int = 2
    decoder_heads: int = 8
    decoder_ffn_dim: int = 2048
    max_headgroup_length: int = 64
    graph: GraphConfig = field(default_factory=GraphConfig)


@dataclass
class RetrievalConfig:
    radius: int = 2
    fp_size: int = 2048
    max_top_k: int = 10
    minimum_similarity: float = 0.4


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 2
    num_workers: int = 0
    cpu_threads: int = 2
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    lambda_arch: float = 1.0
    lambda_chain: float = 1.0
    lambda_head: float = 1.0


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str) -> ExperimentConfig:
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ExperimentConfig()), OmegaConf.load(path))
    )
    if config.model.encoder not in {"graph", "fingerprint"}:
        raise ValueError("encoder must be graph or fingerprint")
    if config.model.structure_dim != 256 or config.model.d_model != 512:
        raise ValueError("v0.1 fixes structure_dim=256 and d_model=512")
    if config.model.graph.pooling != "sum":
        raise ValueError("v0.1 uses sum pooling")
    if config.retrieval.fp_size != 2048 or config.retrieval.radius != 2:
        raise ValueError("v0.1 uses Morgan radius=2, fp_size=2048")
    if not 1 <= config.retrieval.max_top_k <= 10:
        raise ValueError("max_top_k must be 1..10")
    return config
