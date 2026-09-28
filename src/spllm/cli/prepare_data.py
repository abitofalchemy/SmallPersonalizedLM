"""Tokenize personal text files into train/val token-id binaries.

Usage:
    spllm-prepare --input data/raw --output data/processed
"""

import argparse
from pathlib import Path

import numpy as np
import tiktoken

TEXT_SUFFIXES = {".txt", ".md"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw", help="Directory of .txt/.md files")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    files = sorted(p for p in Path(args.input).rglob("*") if p.suffix.lower() in TEXT_SUFFIXES)
    if not files:
        raise SystemExit(f"No .txt or .md files found in {args.input}")

    tokenizer = tiktoken.get_encoding("gpt2")
    ids: list[int] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        ids.extend(tokenizer.encode_ordinary(text))
        ids.append(tokenizer.eot_token)  # <|endoftext|> separates documents

    arr = np.array(ids, dtype=np.uint16)  # GPT-2 vocab (50,257) fits in uint16
    split = int(len(arr) * (1 - args.val_ratio))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    arr[:split].tofile(out / "train.bin")
    arr[split:].tofile(out / "val.bin")

    print(f"Files: {len(files)} | tokens: {len(arr):,} | train: {split:,} | val: {len(arr) - split:,}")
    print(f"Wrote {out / 'train.bin'} and {out / 'val.bin'}")


if __name__ == "__main__":
    main()
