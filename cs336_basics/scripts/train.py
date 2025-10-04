import os
import json
import time
import torch
import wandb
import random
import argparse
import numpy as np
from array import array
from pathlib import Path
from jaxtyping import Float
from cs336_basics.train.adam_w import AdamW
from cs336_basics.train.data_loader import load_data
from cs336_basics.transformer.transformer_lm import TransformerLM
from cs336_basics.train.gradient_clipping import gradient_clipping
from cs336_basics.train.cross_entropy_loss import cross_entropy_loss
from cs336_basics.train.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.train.lr_cosine_schedule import lr_cosine_annealing_schedule


def get_dataset(
        train_dataset_path: str | os.PathLike, 
        meta_path: str | os.PathLike):
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        dtype = np.dtype(meta["dtype"])

        return np.memmap(
            train_dataset_path, 
            dtype=dtype, 
            mode='r', 
            shape=(meta.get("total_tokens", 0),)), dtype


def validate_model(
        step,
        model,
        eval_steps,
        val_dataset,
        batch_size, 
        context_length,
        device,
        start_time,
        run: wandb.Run):
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(eval_steps):
            vx, vy = load_data(
                dataset=val_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device)
            
    
            v_logits = model(vx)
            v_loss = cross_entropy_loss(v_logits.reshape(-1, v_logits.size(-1)), vy.reshape(-1))
            val_losses.append(v_loss)
        
    mean_val_loss = torch.stack(val_losses).mean().item()
    
    print(f"[step {step}] val_loss={mean_val_loss:.4f}", flush=True)

    run.log({
        "step": step,
        "val/loss": mean_val_loss,
        "time/wallclock": time.time() - start_time,
    }, step=step)
    model.train()


def train(
        seed: int,
        wandb_mode: str,
        train_dataset_path: str | os.PathLike,
        train_meta_path: str | os.PathLike,
        val_dataset_path: str | os.PathLike,
        val_meta_path: str | os.PathLike,
        ckpt_path: str | os.PathLike,
        ckpt_interval: int,
        log_interval: int,
        eval_interval: int,
        batch_size: int,
        context_length: int,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        num_steps: int,
        eval_steps: int,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        max_l2_norm: float,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int):

    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    train_dataset, train_dtype = get_dataset(train_dataset_path, train_meta_path)
    val_dataset, val_dtype = get_dataset(val_dataset_path, val_meta_path)

    assert train_dtype == val_dtype, f"train_dtype {train_dtype} != val_dtype {val_dtype}"

    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
    optimizer = AdamW(model.parameters(), max_learning_rate, (beta1, beta2), eps, weight_decay)

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume training if checkpoint exists.
    start_step = 0
    resume_step = -1
    if ckpt_path.exists():
        resume_step = load_checkpoint(ckpt_path, model, optimizer)
        start_step = resume_step + 1
        if start_step >= num_steps:
            print(f"Checkpoint already reached step {resume_step}. Nothing to do.")
        else:
            print(f"Resuming from checkpoint at step {start_step}")

    # model parameter number
    param_cnt = sum(p.numel() for p in model.parameters())

    wandb_id_path = ckpt_path.with_suffix(ckpt_path.suffix + ".wandb_id")
    wandb_resume_kwargs: dict[str, str] = {}
    if start_step > 0 and wandb_id_path.exists():
        wandb_run_id = wandb_id_path.read_text(encoding="utf-8").strip()
        if wandb_run_id:
            wandb_resume_kwargs = {"id": wandb_run_id, "resume": "allow"}

    run = wandb.init(
        project="cs336-assign1",
        name="train_transformer_lm",
        mode=wandb_mode,
        config = {
            "param_cnt": param_cnt,
            "batch_size": batch_size,
            "context_length": context_length,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
            "num_steps": num_steps,
            "eval_steps": eval_steps,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
            "max_l2_norm": max_l2_norm,
            "max_learning_rate": max_learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_iters": warmup_iters,
            "cosine_cycle_iters": cosine_cycle_iters,
        },
        **wandb_resume_kwargs,
    )

    # Persist run id so future resumes can attach to the same W&B run.
    wandb_id_path.write_text(run.id, encoding="utf-8")

    if start_step >= num_steps:
        if resume_step >= 0:
            validate_model(resume_step, model, eval_steps, val_dataset, batch_size, context_length, device, start_time, run)
        run.finish()
        return

    for step in range(start_step, num_steps):
        # Sample dataset
        x, y = load_data(
            dataset=train_dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device)

        model.train()
        optimizer.zero_grad()

        # Run model
        logits = model(x)

        if torch.isnan(logits).any():
            raise RuntimeError("NaN detected in logits")

        # Calculate loss and gradient
        loss = cross_entropy_loss(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), max_l2_norm)

        # Learning rate scheduling
        lr_t = lr_cosine_annealing_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr_t

        # Optimizer
        optimizer.step()

        if step % ckpt_interval == 0:
            save_checkpoint(model, optimizer, step, ckpt_path)

        if step % log_interval == 0:
            print(f"[step {step}] train_loss={loss.item():.4f} lr={lr_t:.2e}", flush=True)
            run.log({
                "step": step,
                "train/loss": loss.item(),
                "train/lr": lr_t,
                "time/wallclock": time.time() - start_time,
            }, step=step)

        if step % eval_interval == 0:
            validate_model(step, model, eval_steps, val_dataset, batch_size, context_length, device, start_time, run)

    save_checkpoint(model, optimizer, step, ckpt_path)
    validate_model(step, model, eval_steps, val_dataset, batch_size, context_length, device, start_time, run)
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--train_dataset_path", type=Path, required=True)
    parser.add_argument("--train_meta_path", type=Path, required=True)
    parser.add_argument("--val_dataset_path", type=Path, required=True)
    parser.add_argument("--val_meta_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--ckpt_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    parser.add_argument("--max_learning_rate", type=float, default=2e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--cosine_cycle_iters", type=int, default=40000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    train(**vars(args))


if __name__ == "__main__":
    main()
