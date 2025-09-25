import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange
from .rms_norm import RMSNorm
from .positionwise_feedforward import SwiGLU
from .multi_head_self_attention import MultiHeadSelfAttention


class TransformerBlock:

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 weights: dict[str, Tensor],
                 ):
    
        q_proj_weight = weights['attn.q_proj.weight']
        k_proj_weight = weights['attn.k_proj.weight']
        v_proj_weight = weights['attn.v_proj.weight']
        o_proj_weight= weights['attn.output_proj.weight']

        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            max_seq_len,
            theta
        )
        
        self.attention.load_state_dict({
            'q_proj_weight': q_proj_weight,
            'k_proj_weight': k_proj_weight,
            'v_proj_weight': v_proj_weight,
            'o_proj_weight': o_proj_weight,
        })

        self.swiglu = SwiGLU(d_model, d_ff)

        self.swiglu.load_state_dict({
            'W1': weights['ffn.w1.weight'],
            'W2': weights['ffn.w2.weight'],
            'W3': weights['ffn.w3.weight'],
        })

        self.rms_norm_attention = RMSNorm(d_model)
        self.rms_norm_attention.load_state_dict({ 'g': weights['ln1.weight'] })

        self.rms_norm_ffn = RMSNorm(d_model)
        self.rms_norm_ffn.load_state_dict({ 'g': weights['ln2.weight'] })


    def forward(self,
                in_features: Float[Tensor, " batch sequence_length d_model"],
                )-> Float[Tensor, " batch sequence_length d_model"]:
        
        # Get token position (batch_size sequence_length) from 0 ~ seq_len-1 for each batch
        token_positions = self.get_token_positions(in_features)

        # attn = x + mh_attn(norm(x))
        after_attention = in_features + self.attention.forward(self.rms_norm_attention.forward(in_features), token_positions)
        
        # output = attn + ffn(norm(attn))
        after_ffn = self.swiglu.forward(self.rms_norm_ffn.forward(after_attention))
        return after_attention + after_ffn


    def get_token_positions(self, 
                            in_features: Float[Tensor, " batch sequence_length d_model"]
                            ) -> Float[Tensor, " ... sequence_length"]:
        batch_size, seq_len = in_features.size(0), in_features.size(-2)
        token_positions = torch.arange(seq_len)
        # Add a new dimension in dim=0, then expand dim=0 to batch_size, keep the second dim unchange
        return token_positions.unsqueeze(0).expand(batch_size, -1)
