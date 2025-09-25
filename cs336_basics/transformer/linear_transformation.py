import math
import torch
from torch import Tensor
from einops import einsum
from jaxtyping import Float


class Linear(torch.nn.Module):

    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        
        super().__init__()

        self.W = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.W, 
            mean = 0, 
            std = std,
            a = -3 * std,
            b = 3 * std)


    def forward(self, x: Float[Tensor, " ... d_in"],) -> Float[Tensor, "... d_out"]:
        return x @ self.W.T  # pytorch row-major
        # return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
