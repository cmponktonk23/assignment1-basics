import os
import torch
import argparse
from pathlib import Path
from cs336_basics.transformer.softmax import softmax
from cs336_basics.train.checkpoint import load_checkpoint
from cs336_basics.bpe.bpe_tokenizer import BPETokenizer
from cs336_basics.transformer.transformer_lm import TransformerLM


def decode(
        prompt: str,
        max_token_num: int,
        temperature: float,
        top_p: float,
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        ckpt_path: str | os.PathLike,
        context_length: int,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        special_tokens: list[str] = ["<|endoftext|>"]):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )

    x = tokenizer.encode(prompt)
    x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)

    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
    load_checkpoint(ckpt_path, model)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_token_num):
            logits = model(x).squeeze(0)[-1, :]

            # temperature scaling
            if temperature <= 0:
                raise ValueError("temperature should greater than 0")
            
            probs = softmax(logits / temperature, -1)

            # top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)

            cutoff = torch.searchsorted(cum_probs, top_p)
            mask = torch.arange(probs.size(-1), device=device) <= cutoff

            truncated_probs = sorted_probs * mask

            denominator = truncated_probs.sum()
            if denominator == 0:
                raise ValueError("top-p too small, sampled prob sum = 0")

            truncated_probs /= denominator

            probs_top_p = torch.zeros_like(probs).scatter(0, sorted_indices, truncated_probs)
            
            # Sample from top-p probability distribution
            next_token_id = torch.multinomial(probs_top_p, num_samples=1)

            # x.append(new_token)
            x = torch.cat([x, next_token_id.unsqueeze(0)], dim=-1)

            # Make sure window size not exceed context_length
            x = x[:, -context_length:]
            
            # Stop when encounter <|endoftext|>
            if next_token_id.item() == tokenizer.vocab_r[b"<|endoftext|>"]:
                break

    print(tokenizer.decode(x.squeeze(0).tolist()))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_token_num", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--vocab_path", type=Path, required=True)
    parser.add_argument("--merges_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    decode(**vars(args))


if __name__ == "__main__":
    main()