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
    
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size, dtype=np.int64)
    
    t1 = torch.empty(batch_size, context_length, dtype=torch.long)
    t2 = torch.empty_like(t1)

    for i, start in enumerate(starts):
        wind = np.asarray(dataset[start : start + 1 + context_length], dtype=np.int64)
        t1[i] = torch.from_numpy(wind[:-1])
        t2[i] = torch.from_numpy(wind[1:])
    
    return t1.to(device, non_blocking=True), t2.to(device, non_blocking=True)
