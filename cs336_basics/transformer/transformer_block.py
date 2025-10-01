import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange
from .rms_norm import RMSNorm
from .positionwise_feedforward import SwiGLU
from .multi_head_self_attention import MultiHeadSelfAttention


class TransformerBlock(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        
        super().__init__()
    
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len,
            theta,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)


    def forward(self,
                in_features: Float[Tensor, " batch sequence_length d_model"],
                )-> Float[Tensor, " batch sequence_length d_model"]:
        
        # Get token position (batch_size sequence_length) from 0 ~ seq_len-1 for each batch
        token_positions = self.get_token_positions(in_features)

        # attn = x + mh_attn(norm(x))
        after_attention = in_features + self.attn.forward(self.ln1.forward(in_features), token_positions)
        
        # output = attn + ffn(norm(attn))
        after_ffn = self.ffn.forward(self.ln2.forward(after_attention))
        return after_attention + after_ffn


    def get_token_positions(self, 
                            in_features: Float[Tensor, " batch sequence_length d_model"]
                            ) -> Float[Tensor, " ... sequence_length"]:
        batch_size, seq_len = in_features.size(0), in_features.size(-2)
        token_positions = torch.arange(seq_len)
        # Add two dimensions `1 1 sequence_length ...`, then broadcast them to `batch_size num_heads sequence_length ...`
        return token_positions
        # Add one dimension `batch_size 1 sequence_length ...`, then broadcast it to `batch_size num_heads sequence_length ...`
        # return token_positions.unsqueeze(0).expand(batch_size, -1)
