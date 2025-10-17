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

        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps


    def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape `(batch_size, *)`.
                The input to apply root mean square layer normalization on.

        Returns:
            FloatTensor of same shape as input.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.square().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        x = x * rms * self.weight
        return x.to(in_dtype)