# Small Personalized Language Model

A small GPT-style language model built **from scratch** in PyTorch, following
Sebastian Raschka's *Build a Large Language Model (From Scratch)* (Manning, 2024),
and trained on a personal text corpus.

**Target hardware:** Mac mini, Apple M4 Pro, 24 GB unified memory (PyTorch `mps` backend).

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [Repository Structure](#repository-structure)
3. [Setup](#setup)
4. [Quickstart](#quickstart)
5. [Model Configurations](#model-configurations)
6. [Personal Data](#personal-data)
7. [Roadmap](#roadmap)
8. [Hardware Notes (M4 Pro)](#hardware-notes-m4-pro)
9. [References](#references)
10. [License](#license)

---

## Project Goals

- Implement every component of a GPT model by hand: tokenization, data loading,
  multi-head causal self-attention, transformer blocks, and the training loop.
- Pretrain a small model on a personal corpus (notes, writing, documents).
- Fine-tune it for personal tasks (instruction following, classification).
- Keep the whole thing small enough to train locally on an M4 Pro.

## Repository Structure

```
SmallPersonalizedLM/
├── README.md               # This file
├── LICENSE
├── pyproject.toml          # Package metadata, dependencies, CLI entry points
├── uv.lock                 # Locked dependency versions (managed by uv)
├── .python-version         # Python version uv uses for this project
├── configs/                # Model and training hyperparameters (YAML)
│   ├── tiny.yaml           #   fast smoke-test model
│   └── gpt2_small.yaml     #   GPT-2 small (124M) architecture
├── data/
│   ├── raw/                # Your personal text (.txt/.md) — git-ignored
│   └── processed/          # Tokenized datasets — git-ignored
├── checkpoints/            # Saved model weights — git-ignored
├── docs/
│   └── roadmap.md          # Book chapter → project milestone mapping
├── notebooks/              # Exploratory notebooks, one per book chapter
├── src/spllm/              # The installable package
│   ├── __init__.py
│   ├── cli/                # Command-line tools (installed as spllm-* commands)
│   │   ├── download_sample.py  # spllm-download-sample: fetch "The Verdict"
│   │   ├── prepare_data.py     # spllm-prepare: raw text → token ids (.bin)
│   │   ├── train.py            # spllm-train: pretraining loop
│   │   └── generate.py         # spllm-generate: text from a checkpoint
│   ├── config.py           # GPTConfig dataclass + YAML loader
│   ├── data.py             # Sliding-window dataset & dataloaders
│   ├── model.py            # GPT model (attention, blocks, LM head)
│   ├── generate.py         # Sampling (temperature, top-k)
│   └── utils.py            # Device selection, seeding, checkpoint I/O
└── tests/
    └── test_model.py       # Shape / sanity tests
```

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/).
Install uv first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: brew install uv
```

**Development (clone and work on the code):**

```bash
git clone https://github.com/abitofalchemy/SmallPersonalizedLM.git
cd SmallPersonalizedLM
uv sync                        # creates .venv, installs spllm (editable) + dev tools
uv sync --group notebooks      # optional: JupyterLab + matplotlib
```

**Install as a package only (no clone needed):**

```bash
uv pip install "git+https://github.com/abitofalchemy/SmallPersonalizedLM.git"
# or install just the command-line tools, globally:
uv tool install "git+https://github.com/abitofalchemy/SmallPersonalizedLM.git"
```

Plain `pip install .` works too; the package uses standard `pyproject.toml` metadata.

Check that PyTorch sees the Apple GPU:

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

## Quickstart

```bash
# 1. Put your .txt / .md files in data/raw/
#    (or grab the book's sample text for a first test run)
uv run spllm-download-sample

# 2. Tokenize them
uv run spllm-prepare --input data/raw --output data/processed

# 3. Train (start with the tiny config to verify everything works)
uv run spllm-train --config configs/tiny.yaml

# 4. Generate text
uv run spllm-generate --checkpoint checkpoints/tiny/model.pt \
    --prompt "Today I want to" --max-new-tokens 50

# Run tests
uv run pytest
```

If you activate the environment (`source .venv/bin/activate`), you can drop
the `uv run` prefix.

## Model Configurations

| Config            | Layers | Heads | Embedding | Context | ~Params | Purpose                 |
|-------------------|:------:|:-----:|:---------:|:-------:|:-------:|-------------------------|
| `tiny.yaml`       | 4      | 4     | 256       | 256     | ~30M    | Smoke tests, debugging  |
| `gpt2_small.yaml` | 12     | 12    | 768       | 256     | ~124M   | Main personalized model |

Both use the GPT-2 BPE tokenizer (`tiktoken`, vocab 50,257). Most of the
parameter count in small configs comes from the token embedding and output head.

## Personal Data

- Place source text in `data/raw/`. It is **git-ignored** so private writing is
  never pushed to GitHub.
- Tokenized outputs in `data/processed/` and weights in `checkpoints/` are also
  ignored.
- Before making the repository public, double-check `git status` for anything
  personal.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the full plan. Summary:

- [x] Repository structure
- [ ] Ch. 2 — Tokenization & data loading
- [ ] Ch. 3 — Attention mechanisms
- [ ] Ch. 4 — GPT model architecture
- [ ] Ch. 5 — Pretraining on personal corpus
- [ ] Ch. 6 — Fine-tuning for classification
- [ ] Ch. 7 — Instruction fine-tuning

## Hardware Notes (M4 Pro)

- The device is chosen automatically: `mps` → `cuda` → `cpu`.
- 24 GB unified memory comfortably fits the 124M model with batch size 8–16
  at context length 256. Reduce `batch_size` if you see memory pressure.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if an op is not yet implemented on MPS.

## References

- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024.
- Official code: <https://github.com/rasbt/LLMs-from-scratch>
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.
- Vaswani et al., *Attention Is All You Need*, 2017.

## License

Released under the [MIT License](LICENSE).
