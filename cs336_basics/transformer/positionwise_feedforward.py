import torch
from einops import einsum
from .linear import Linear


class SwiGLU(torch.nn.Module):

    def __init__(self, 
                 d_model: int, 
                 d_ff: int,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)


    @classmethod
    def silu(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


    def forward(self, x: torch.Tensor):
        xw1 = einsum(x, self.w1.weight, "... d_model, d_ff d_model -> ... d_ff")
        xw3 = einsum(x, self.w3.weight, "... d_model, d_ff d_model -> ... d_ff")
        swiglu = SwiGLU.silu(xw1) * xw3
        return einsum(swiglu, self.w2.weight, "... d_ff, d_model d_ff -> ... d_model")
