"""Generate text from a trained checkpoint.

Usage:
    spllm-generate --checkpoint checkpoints/tiny/model.pt --prompt "Today I"
"""

import argparse

from spllm.generate import generate_text
from spllm.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Today I")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    print(generate_text(
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ))


if __name__ == "__main__":
    main()
