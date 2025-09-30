import torch
import numpy.typing as npt


def load_data(
        dataset: npt.NDArray, 
        batch_size: int, 
        context_length: int, 
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    ds = torch.as_tensor(dataset, device=device)
    t1 = torch.empty(batch_size, context_length, device=device)
    t2 = torch.empty_like(t1)

    max_start = len(ds) - context_length
    starts = torch.randint(0, max_start, (batch_size,), device=device)

    for i, start in enumerate(starts):
        t1[i], t2[i] = ds[start:start + context_length], ds[start + 1:start + 1 + context_length]
    
    return t1, t2
