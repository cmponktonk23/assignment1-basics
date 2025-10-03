import os
import json
import argparse
from array import array
from pathlib import Path
from tqdm import tqdm
from cs336_basics.bpe.bpe_tokenizer import BPETokenizer


def tokenize(
        input_path: str | os.PathLike,
        train_out_path: str | os.PathLike,
        vocab_path: str | os.PathLike, 
        merges_path: str | os.PathLike,
        special_tokens: list[str] = ["<|endoftext|>"]):
    
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )

    train_out_path = Path(train_out_path)
    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    train_meta_path = train_out_path.with_suffix(train_out_path.suffix + ".meta.json")
    train_buf = array("H")
    train_cnt = 0

    # Get file size for progress bar
    file_size = os.path.getsize(input_path)

    with open(input_path, 'r', encoding="utf-8") as fin, \
         open(train_out_path, "wb") as ftrain, \
         open(train_meta_path, "w") as ftrain_meta:

        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Tokenizing")
        last_pos = 0

        for token_id in tokenizer.encode_iterable(fin):
            train_buf.append(token_id)
            if len(train_buf) >= 1_000_000:
                train_cnt += len(train_buf)
                train_buf.tofile(ftrain)
                train_buf = array("H")

                # Update progress bar
                current_pos = fin.tell()
                pbar.update(current_pos - last_pos)
                last_pos = current_pos

        if train_buf:
            train_cnt += len(train_buf)
            train_buf.tofile(ftrain)

        # Final progress update
        current_pos = fin.tell()
        pbar.update(current_pos - last_pos)
        pbar.close()
    
        json.dump({
            "total_tokens": train_cnt,
            "dtype": "uint16",
        }, ftrain_meta, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--train_out_path", type=Path, required=True)
    parser.add_argument("--vocab_path", type=Path, required=True)
    parser.add_argument("--merges_path", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenize(**vars(args))


if __name__ == "__main__":
    main()