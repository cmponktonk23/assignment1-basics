import torch
from torch import Tensor
from jaxtyping import Float


class RMSNorm(torch.nn.Module):

    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        
        super().__init__()

        self.g = torch.nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.eps = eps


    def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * self.g / (x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return x.to(in_dtype)