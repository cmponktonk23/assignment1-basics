import torch
from torch import Tensor
from einops import einsum, rearrange
from jaxtyping import Float, Int
from .rope import RoPE
from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 q_proj_weight: Float[Tensor, " d_k d_in"],
                 k_proj_weight: Float[Tensor, " d_k d_in"],
                 v_proj_weight: Float[Tensor, " d_v d_in"],
                 o_proj_weight: Float[Tensor, " d_model d_v"],
                 max_seq_len: int | None = None,
                 theta: float | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads

        self.q_proj_weight = torch.nn.Parameter(torch.empty_like(q_proj_weight))
        self.k_proj_weight = torch.nn.Parameter(torch.empty_like(k_proj_weight))
        self.v_proj_weight = torch.nn.Parameter(torch.empty_like(v_proj_weight))
        self.o_proj_weight = torch.nn.Parameter(torch.empty_like(o_proj_weight))

        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.head_size, max_seq_len, device)
        

    def forward(self, 
                in_features: Float[Tensor, " ... sequence_length d_in"],
                token_positions: Int[Tensor, " ... sequence_length"] | None = None,
                ) -> Float[Tensor, " ... sequence_length d_out"]:
        
        weight_combine = torch.cat([self.q_proj_weight, self.k_proj_weight, self.v_proj_weight], dim=0)
        QKV = einsum(in_features, weight_combine, "... sequence_length d_in, d_qkv d_in -> ... sequence_length d_qkv")

        Q, K, V = QKV.split((self.q_proj_weight.size(-1), self.k_proj_weight.size(-1), self.v_proj_weight.size(-1)), dim=-1)

        Qh = rearrange(Q, '... sequence_length (n_head d_k) -> ... n_head sequence_length d_k', n_head=self.num_heads)
        Kh = rearrange(K, '... sequence_length (n_head d_k) -> ... n_head sequence_length d_k', n_head=self.num_heads)
        Vh = rearrange(V, '... sequence_length (n_head d_v) -> ... n_head sequence_length d_v', n_head=self.num_heads)

        if self.rope is not None:
            Qh = self.rope.forward(Qh, token_positions)
            Kh = self.rope.forward(Kh, token_positions)

        seq_len = in_features.size(-2)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()

        attention = ScaledDotProductAttention(Qh, Kh, Vh, mask).forward()
        attention = rearrange(attention, '... n_head sequence_length d_v -> ... sequence_length (n_head d_v)', n_head=self.num_heads)
        
        return einsum(attention, self.o_proj_weight, "... sequence_length d_v, d_model d_v -> ... sequence_length d_model")
