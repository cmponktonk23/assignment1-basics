import os
import torch
import typing


def save_checkpoint(
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        iteration: int, 
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(ckpt, out)


def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer):

    ckpt = torch.load(src)
    iteration = 0

    if ckpt:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        iteration = ckpt["iteration"]

    return iteration
