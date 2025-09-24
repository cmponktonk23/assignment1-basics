import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum


class RoPE(torch.nn.Module):

    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()

        positions = torch.arange(max_seq_len)

        half_d = d_k >> 1
        inv_freq = torch.pow(theta, -((2 * torch.arange(1, half_d + 1) - 2) / d_k))
        
        angles = positions[:, None] * inv_freq[None, :]
        
        self.register_buffer("sin", angles.sin(), persistent=False)
        self.register_buffer("cos", angles.cos(), persistent=False)


    def forward(self,
                x: Float[Tensor, " ... sequence_length d_k"],
                token_positions: Int[Tensor, " ... sequence_length"]) -> torch.Tensor:
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = rot_even
        x_rot[..., 1::2] = rot_odd

        return x_rot