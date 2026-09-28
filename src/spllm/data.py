"""Sliding-window next-token dataset (Raschka, ch. 2)."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    """Chunks a 1-D array of token ids into (input, target) pairs shifted by one."""

    def __init__(self, token_ids, max_length: int, stride: int):
        ids = torch.as_tensor(np.asarray(token_ids, dtype=np.int64))
        self.inputs = []
        self.targets = []
        for i in range(0, len(ids) - max_length, stride):
            self.inputs.append(ids[i : i + max_length])
            self.targets.append(ids[i + 1 : i + max_length + 1])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_tokens(path: str | Path) -> np.ndarray:
    """Load token ids written by `spllm-prepare`."""
    return np.fromfile(path, dtype=np.uint16)


def create_dataloader(
    token_ids,
    batch_size: int = 8,
    max_length: int = 256,
    stride: int | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TokenDataset(token_ids, max_length, stride or max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
