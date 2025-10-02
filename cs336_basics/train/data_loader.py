import torch
import numpy as np
import numpy.typing as npt


def load_data(
        dataset: npt.NDArray, 
        batch_size: int, 
        context_length: int, 
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    if len(dataset) <= context_length:
        raise ValueError("dataset too short for given context_length")
    
    data = torch.as_tensor(dataset, dtype=torch.long)
    starts = torch.randint(0, len(data) - context_length, (batch_size,))
    offsets = torch.arange(context_length + 1)
    seqs = data[starts.unsqueeze(1) + offsets]
    x = seqs[:, :-1].to(device, non_blocking=True)
    y = seqs[:, 1:].to(device, non_blocking=True)
    
    return x, y
