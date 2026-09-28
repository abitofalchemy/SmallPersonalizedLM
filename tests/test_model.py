import numpy as np
import torch

from spllm import GPTConfig, GPTModel
from spllm.data import create_dataloader
from spllm.generate import generate

TINY = GPTConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=2, drop_rate=0.0)


def test_forward_shape():
    model = GPTModel(TINY)
    x = torch.randint(0, TINY.vocab_size, (2, 10))
    assert model(x).shape == (2, 10, TINY.vocab_size)


def test_attention_is_causal():
    torch.manual_seed(0)
    model = GPTModel(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (1, 8))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % TINY.vocab_size  # change only the last token
    out_x, out_y = model(x), model(y)
    assert torch.allclose(out_x[:, :-1], out_y[:, :-1], atol=1e-5)


def test_generate_length():
    model = GPTModel(TINY).eval()
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = generate(model, idx, max_new_tokens=5, context_size=TINY.context_length)
    assert out.shape == (1, 8)


def test_dataloader_targets_are_shifted():
    ids = np.arange(100)
    loader = create_dataloader(ids, batch_size=2, max_length=8, shuffle=False)
    x, y = next(iter(loader))
    assert torch.equal(y, x + 1)
