"""spllm — Small Personalized Language Model built from scratch."""

from spllm.config import GPTConfig, load_config
from spllm.model import GPTModel

__all__ = ["GPTConfig", "GPTModel", "load_config"]
__version__ = "0.1.0"
