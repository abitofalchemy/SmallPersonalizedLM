"""Model and training configuration."""

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False


@dataclass
class TrainConfig:
    data_dir: str = "data/processed"
    out_dir: str = "checkpoints/default"
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 4e-4
    weight_decay: float = 0.1
    eval_freq: int = 100
    eval_iters: int = 20
    seed: int = 123


@dataclass
class Config:
    model: GPTConfig = field(default_factory=GPTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls, values: dict | None):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> Config:
    """Load a YAML file with optional `model:` and `train:` sections."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        model=_build(GPTConfig, raw.get("model")),
        train=_build(TrainConfig, raw.get("train")),
    )
