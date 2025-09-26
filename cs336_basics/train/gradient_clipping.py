import torch
from typing import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    # Calculate global l2-norm(g) = sqrt(g1^2 + g2^2 + ... + gn^2) 
    g = torch.cat([p.grad for p in parameters if p.grad is not None])
    g_norm2 = torch.norm(g, p=2)

    if g_norm2 >= max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad *= max_l2_norm / (g_norm2 + 1e-6)