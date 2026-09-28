# Roadmap

Milestones follow the chapters of Sebastian Raschka's
*Build a Large Language Model (From Scratch)*.

| Ch. | Topic                           | Project deliverable                                   | Code                       | Status |
|:---:|---------------------------------|-------------------------------------------------------|----------------------------|:------:|
| 1   | Understanding LLMs              | Notes in `notebooks/ch01_overview.ipynb`              | —                          | ☐      |
| 2   | Working with text data          | BPE tokenization + sliding-window dataset             | `src/spllm/data.py`, `src/spllm/cli/prepare_data.py` | ☐ |
| 3   | Coding attention mechanisms     | Causal multi-head attention                           | `src/spllm/model.py`       | ☐      |
| 4   | Implementing a GPT model        | Full GPT architecture, parameter counting             | `src/spllm/model.py`       | ☐      |
| 5   | Pretraining on unlabeled data   | Train on personal corpus; loss curves; sampling       | `src/spllm/cli/train.py`, `src/spllm/generate.py` | ☐ |
| 6   | Fine-tuning for classification  | Classify personal notes (e.g., topic / mood)          | `src/spllm/finetune_cls.py` (todo) | ☐ |
| 7   | Fine-tuning to follow instructions | Personal assistant on custom instruction data      | `src/spllm/finetune_instruct.py` (todo) | ☐ |

## Stretch goals

- Train a custom BPE tokenizer on the personal corpus.
- Learning-rate warmup + cosine decay and gradient clipping (book appendix D).
- LoRA fine-tuning (book appendix E).
- Load OpenAI GPT-2 weights as a starting point and compare against from-scratch training.
- Track experiments (loss curves, samples) in `runs/`.
