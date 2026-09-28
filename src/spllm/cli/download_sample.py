"""Download the book's sample text ("The Verdict") for a first test run.

Usage:
    spllm-download-sample [--output data/raw]
"""

import argparse
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw", help="Directory to save into")
    args = parser.parse_args()

    file_path = Path(args.output) / "the-verdict.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, file_path)
    print(f"Saved {file_path}")


if __name__ == "__main__":
    main()
