"""Tokenizing "The Verdict" (Raschka, ch. 2): from word-level tokens to byte pair encoding.

Steps:
1. Download the raw text once into data/raw/.
2. Preprocess: normalize the text and split it into words and punctuation.
3. SimpleTokenizer: word-level vocabulary with <|unk|> and <|endoftext|> tokens.
4. BPETokenizer: byte-level byte pair encoding trained from scratch on the text,
   using GPT-4-style pre-tokenization. Any string can be encoded (no <|unk|>).
5. Compare with tiktoken's pretrained BPE encodings: gpt2 (50k vocab, used by
   the model) and o200k_base (GPT-4o, 200k vocab).
6. Build sliding-window input/target pairs for next-token prediction.

Run:
    uv run python src/spllm/tst-the-verdict.py
"""

import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import regex
import tiktoken

from spllm.data import create_dataloader

URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)
RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "the-verdict.txt"

EOT = "<|endoftext|>"
UNK = "<|unk|>"


# --- 1. Raw text -------------------------------------------------------------

def load_raw_text(path: Path = RAW_PATH) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path.read_text(encoding="utf-8")


# --- 2. Preprocessing --------------------------------------------------------

SPLIT_RE = re.compile(r'([,.:;?_!"()\']|--|\s)')


def normalize(text: str) -> str:
    """Unicode NFC, Unix newlines, and straight quotes."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    return text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'}))


def preprocess(text: str) -> list[str]:
    """Split into word and punctuation tokens, dropping whitespace."""
    return [t for t in (s.strip() for s in SPLIT_RE.split(text)) if t]


# --- 3. Word-level tokenizer -------------------------------------------------

class SimpleTokenizer:
    """Maps each distinct word/punctuation token to an id; unknown words become <|unk|>."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    @classmethod
    def from_text(cls, text: str) -> "SimpleTokenizer":
        words = sorted(set(preprocess(normalize(text)))) + [EOT, UNK]
        return cls({w: i for i, w in enumerate(words)})

    def encode(self, text: str) -> list[int]:
        tokens = preprocess(normalize(text))
        return [self.str_to_int.get(t, self.str_to_int[UNK]) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)


# --- 4. Byte pair encoding (trained from scratch) ----------------------------

# GPT-4 (cl100k) pre-tokenization: merges never cross word, number, or punctuation boundaries.
PRETOKENIZE_RE = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def _merge(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """Byte-level BPE: starts from 256 byte tokens and learns merges of frequent pairs."""

    def __init__(self, merges: dict[tuple[int, int], int], special_tokens: dict[str, int]):
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in merges.items():  # dicts keep insertion (= merge) order
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        for token, idx in special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")
        self._special_re = re.compile("(" + "|".join(map(re.escape, special_tokens)) + ")") if special_tokens else None

    @classmethod
    def train(cls, text: str, vocab_size: int, special_tokens=(EOT,)) -> "BPETokenizer":
        num_merges = vocab_size - 256 - len(special_tokens)
        assert num_merges >= 0, "vocab_size too small"

        # Count each distinct pre-token once, weighted by frequency (much faster than a flat list).
        chunks = Counter(tuple(c.encode("utf-8")) for c in PRETOKENIZE_RE.findall(normalize(text)))
        merges: dict[tuple[int, int], int] = {}
        for new_id in range(256, 256 + num_merges):
            stats = Counter()
            for ids, freq in chunks.items():
                for pair in zip(ids, ids[1:]):
                    stats[pair] += freq
            if not stats:
                break  # every pre-token is already a single token
            pair = max(stats, key=stats.get)
            merges[pair] = new_id
            chunks = Counter({_merge(ids, pair, new_id): freq for ids, freq in chunks.items()})

        first_special = 256 + len(merges)
        return cls(merges, {t: first_special + i for i, t in enumerate(special_tokens)})

    def _encode_chunk(self, data: bytes) -> list[int]:
        ids = tuple(data)
        while len(ids) >= 2:
            # Apply the earliest-learned merge available, exactly as in training.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return list(ids)

    def encode(self, text: str) -> list[int]:
        parts = self._special_re.split(text) if self._special_re else [text]
        ids: list[int] = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                for chunk in PRETOKENIZE_RE.findall(normalize(part)):
                    ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# --- Walkthrough -------------------------------------------------------------

def main():
    raw = load_raw_text()
    text = normalize(raw)
    print(f"Raw text: {len(raw):,} characters\n{text[:99]!r}\n")

    tokens = preprocess(text)
    print(f"Preprocessed: {len(tokens):,} tokens\n{tokens[:30]}\n")

    sample = f'"It\'s the last he painted, you know," Mrs. Gisburn said. {EOT} Hello, do you like tea?'

    simple = SimpleTokenizer.from_text(raw)
    ids = simple.encode(sample)
    print(f"SimpleTokenizer — vocab {len(simple.str_to_int):,}")
    print(f"  ids:     {ids}")
    print(f"  decoded: {simple.decode(ids)}  (unseen words became {UNK})\n")

    bpe = BPETokenizer.train(raw, vocab_size=1000)
    ids = bpe.encode(sample)
    assert bpe.decode(ids) == normalize(sample), "BPE round trip failed"
    corpus_ids = bpe.encode(raw)
    print(f"BPETokenizer (trained here) — vocab {bpe.vocab_size:,}")
    print(f"  pieces:  {[bpe.decode([i]) for i in ids]}")
    print(f"  corpus:  {len(corpus_ids):,} tokens, {len(text.encode()) / len(corpus_ids):.2f} bytes/token")
    print(f"  longest learned tokens: {sorted((bpe.vocab[i] for i in bpe.merges.values()), key=len)[-5:]}\n")

    for name in ("gpt2", "o200k_base"):
        enc = tiktoken.get_encoding(name)
        ids = enc.encode(sample, allowed_special={EOT})
        assert enc.decode(ids) == sample
        corpus_ids = enc.encode(text, allowed_special={EOT})
        print(f"tiktoken {name} — vocab {enc.n_vocab:,}")
        print(f"  pieces:  {[enc.decode([i]) for i in ids]}")
        print(f"  corpus:  {len(corpus_ids):,} tokens, {len(text.encode()) / len(corpus_ids):.2f} bytes/token\n")

    # The model uses the gpt2 encoding: turn the corpus into (input, target) windows.
    gpt2 = tiktoken.get_encoding("gpt2")
    loader = create_dataloader(gpt2.encode(text), batch_size=2, max_length=8, stride=8, shuffle=False)
    x, y = next(iter(loader))
    print("Sliding-window pairs (targets are inputs shifted by one token):")
    print(f"  inputs:  {x[0].tolist()} -> {gpt2.decode(x[0].tolist())!r}")
    print(f"  targets: {y[0].tolist()} -> {gpt2.decode(y[0].tolist())!r}")


if __name__ == "__main__":
    main()
