import os
import json
import argparse
from pathlib import Path
from cs336_basics.bpe.train_bpe import train_bpe, b2u, u2b


def train(
        input_path: str | os.PathLike,
        out_dir: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str] = ["<|endoftext|>",]):
    
    vocab, merges = train_bpe(
        input_path = input_path,
        vocab_size = vocab_size, 
        special_tokens = special_tokens)

    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.txt"
    
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(idx): b2u(b) for idx, b in vocab.items()},
                  f,
                  ensure_ascii=False,
                  indent=2)
        
    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{b2u(left)} {b2u(right)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    train(args.input_path, args.out_dir, args.vocab_size)


if __name__ == "__main__":
    main()