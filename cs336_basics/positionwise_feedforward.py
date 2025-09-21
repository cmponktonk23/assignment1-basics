import torch
from einops import einsum


class SwiGLU(torch.nn.Module):

    def __init__(self, 
                 d_model: int, 
                 d_ff: int,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = torch.nn.Parameter(torch.empty(d_model, d_ff))
        self.W3 = torch.nn.Parameter(torch.empty(d_ff, d_model))


    @classmethod
    def silu(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


    def forward(self, x: torch.Tensor):
        xw1 = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        xw3 = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        swiglu = SwiGLU.silu(xw1) * xw3
        return einsum(swiglu, self.W2, "... d_ff, d_model d_ff -> ... d_model")
