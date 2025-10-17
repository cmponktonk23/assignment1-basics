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
    
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size, dtype=np.int64)  # (batch_size,)
    offsets = np.arange(context_length + 1, dtype=np.int64)      # (context_length + 1,)
    idx = starts[:, None] + offsets[None, :]                     # broadcast to [batch_size, context_length + 1]
    seqs = np.asarray(dataset[idx], dtype=np.int64)              # [batch_size, context_length + 1]
    
    x = torch.from_numpy(seqs[:, :-1]).to(device, non_blocking=True)
    y = torch.from_numpy(seqs[:, 1:]).to(device, non_blocking=True)
    
    return x, y
