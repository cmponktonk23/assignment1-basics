import math
from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum
from .softmax import softmax


def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"],
                                 K: Float[Tensor, " ... keys d_k"],
                                 V: Float[Tensor, " ... values d_v"],
                                 mask: Bool[Tensor, " ... queries keys"] | None = None,
                                 ) -> Float[Tensor, " ... queries d_v"]:

    # QK/sqrt(d_k)    
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    attn_logits = QK / math.sqrt(Q.shape[-1])

    # causal mask
    masked_logits = attn_logits.masked_fill(~mask, float("-inf"))
    
    # softmax
    attn_weights = softmax(masked_logits, -1)
    
    # weight * V
    return einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
