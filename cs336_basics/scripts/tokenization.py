import os
import json
import argparse
from array import array
from pathlib import Path
from cs336_basics.bpe.bpe_tokenizer import BPETokenizer


def tokenize(
        # train_ratio: float,
        input_path: str | os.PathLike,
        train_out_path: str | os.PathLike,
        # val_out_path: str | os.PathLike,
        vocab_path: str | os.PathLike, 
        merges_path: str | os.PathLike,
        special_tokens: list[str] = ["<|endoftext|>"]):
    
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )

    # First pass count total tokens number
    # total_tokens = 0
    # with open(input_path, encoding="utf-8") as f:
    #     for _ in tokenizer.encode_iterable(f):
    #         total_tokens += 1

    # train_limit = int(total_tokens * train_ratio)

    train_out_path = Path(train_out_path)
    # val_out_path = Path(val_out_path)
    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    # val_out_path.parent.mkdir(parents=True, exist_ok=True)
    train_meta_path = train_out_path.with_suffix(train_out_path.suffix + ".meta.json")
    # val_meta_path = val_out_path.with_suffix(val_out_path.suffix + ".meta.json")
    train_buf = array("H")
    # val_buf = array("H")
    train_cnt = 0
    # val_cnt = 0

    with open(input_path, 'r', encoding="utf-8") as fin, \
         open(train_out_path, "wb") as ftrain, \
         open(train_meta_path, "w") as ftrain_meta:
        #  open(val_out_path, "wb") as fval, \
        #  open(val_meta_path, "w") as fval_meta:
        
        # idx = 0
        for token_id in tokenizer.encode_iterable(fin):
            # if idx < train_limit:
            train_buf.append(token_id)
            if len(train_buf) >= 1_000_000:
                train_cnt += len(train_buf)
                train_buf.tofile(ftrain)
                train_buf = array("H")
            # else:
            #     val_buf.append(token_id)
            #     if len(val_buf) >= 1_000_000:
            #         val_cnt += len(val_buf)
            #         val_buf.tofile(fval)
            #         val_buf = array("H")
            # idx += 1

        if train_buf:
            train_cnt += len(train_buf)
            train_buf.tofile(ftrain)
        # if val_buf:
        #     val_cnt += len(val_buf)
        #     val_buf.tofile(fval)
    
        json.dump({
            "total_tokens": train_cnt,
            "dtype": "uint16",
        }, ftrain_meta, ensure_ascii=False, indent=2)

        # json.dump({
        #     "total_tokens": val_cnt,
        #     "dtype": "uint16",
        # }, fval_meta, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--train_out_path", type=Path, required=True)
    # parser.add_argument("--val_out_path", type=Path, required=True)
    parser.add_argument("--vocab_path", type=Path, required=True)
    parser.add_argument("--merges_path", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenize(**vars(args))


if __name__ == "__main__":
    main()