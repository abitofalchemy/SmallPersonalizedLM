"""Device selection, seeding, and checkpoint I/O."""

from dataclasses import asdict
from pathlib import Path

import torch

from spllm.config import GPTConfig
from spllm.model import GPTModel


def get_device() -> torch.device:
    """Prefer Apple Silicon GPU (mps), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def save_checkpoint(path: str | Path, model: GPTModel, optimizer=None, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_config": asdict(model.cfg), "model_state_dict": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> GPTModel:
    device = device or get_device()
    state = torch.load(path, map_location=device, weights_only=True)
    model = GPTModel(GPTConfig(**state["model_config"]))
    model.load_state_dict(state["model_state_dict"])
    return model.to(device)
