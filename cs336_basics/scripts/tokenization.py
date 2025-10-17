import os
import json
import argparse
from array import array
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, RLock
from functools import partial
from cs336_basics.bpe.bpe_tokenizer import BPETokenizer
from cs336_basics.bpe.pretokenization_example import find_chunk_boundaries


tqdm.set_lock(RLock())


class LimitedReader:
    def __init__(self, file, start, end):
        self.file = file
        self.current = start
        self.end = end


    def __iter__(self):
        return self
    

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        
        line_bytes = self.file.readline()
        if not line_bytes:
            raise StopIteration
        
        line_len = len(line_bytes)

        if self.current + line_len > self.end:
            remaining = self.end - self.current
            line = line_bytes[:remaining].decode("utf-8", errors="ignore")
            self.current = self.end
        else:
            line = line_bytes.decode("utf-8", errors="ignore")
            self.current += line_len

        return line


def tokenize_test(
        num_processes: int,
        input_path: str | os.PathLike,
        out_path: str | os.PathLike,
        vocab_path: str | os.PathLike, 
        merges_path: str | os.PathLike,
        special_tokens: list[str] = ["<|endoftext|>"]):
    
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )

    with open(input_path, 'r', encoding="utf-8") as fin, \
         open(f"{out_path}.test", 'wb') as fout:
    
        train_buf = array("H")
        token_cnt = 0
        for token_id in tokenizer.encode_iterable(fin):
            train_buf.append(token_id)
            token_cnt += 1

        train_buf.tofile(fout)
        
        meta_path = Path(out_path).with_suffix('.test.meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                "total_tokens": token_cnt,
                "dtype": "uint16",
            }, f, ensure_ascii=False, indent=2)


def tokenize(
        num_processes: int,
        input_path: str | os.PathLike,
        out_path: str | os.PathLike,
        vocab_path: str | os.PathLike, 
        merges_path: str | os.PathLike,
        special_tokens: list[str] = ["<|endoftext|>"]):
    
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )

    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        jobs = [(worker_id, start, end) for worker_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
        worker = partial(tokenization, tokenizer, input_path, out_path)
        
        with Pool(processes=num_processes) as pool:
            token_cnts: list[int] = pool.starmap(worker, jobs)

        print("Start to merge tmp files...")

        # Merge tmp files
        with open(out_path, 'wb') as fout:
            for worker_id, _, _ in jobs:
                tmp_file = Path(f"{out_path}.tmp.{worker_id}")
                with open(tmp_file, 'rb') as fin:
                    fout.write(fin.read())
                tmp_file.unlink()

        # Write metadata
        meta_path = Path(out_path).with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                "total_tokens": sum(token_cnts),
                "dtype": "uint16",
            }, f, ensure_ascii=False, indent=2)


def tokenization(
        tokenizer: BPETokenizer,
        input_path: str | os.PathLike,
        out_path: str | os.PathLike,
        worker_id: int,
        start: int,
        end: int):

    out_path = Path(f"{out_path}.tmp.{worker_id}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'rb') as fin, \
         open(out_path, "wb") as ftrain:
        
        # Read the chunk assigned to current worker
        fin.seek(start)

        def handle_chunk():
            chunk = fin.read(end - start).decode("utf-8", errors="ignore")
            token_ids = tokenizer.encode(chunk)
            token_cnt = len(token_ids)
            train_buf = array("H", token_ids)
            train_buf.tofile(ftrain)
            return token_cnt

        def handle_stream():
            train_buf = array("H")
            token_cnt = 0
            processed = 0
            limited_reader = LimitedReader(fin, start, end)
            with tqdm(total=end-start, unit="B", unit_scale=True, desc=f"Worker {worker_id}", position=worker_id, leave=False, dynamic_ncols=True) as pbar:
                for token_id in tokenizer.encode_iterable(limited_reader):
                    train_buf.append(token_id)
                    if len(train_buf) >= 1_000_000:
                        token_cnt += len(train_buf)
                        train_buf.tofile(ftrain)
                        train_buf = array("H")
                        
                    new_processed = limited_reader.current - start
                    if new_processed > processed:
                        pbar.update(new_processed - processed)
                        processed = new_processed

                if train_buf:
                    token_cnt += len(train_buf)
                    train_buf.tofile(ftrain)

                return token_cnt
    
        return handle_stream()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--vocab_path", type=Path, required=True)
    parser.add_argument("--merges_path", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenize(**vars(args))
    # tokenize_test(**vars(args))


if __name__ == "__main__":
    main()