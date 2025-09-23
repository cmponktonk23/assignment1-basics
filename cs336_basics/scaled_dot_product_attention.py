import math
import torch
from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum
from cs336_basics.softmax import SoftMax


class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self,
                 Q: Float[Tensor, " ... queries d_k"],
                 K: Float[Tensor, " ... keys d_k"],
                 V: Float[Tensor, " ... values d_v"],
                 mask: Bool[Tensor, " ... queries keys"] | None = None,
                 ) -> Float[Tensor, " ... queries d_v"]:
        super().__init__()

        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask


    def forward(self):
        QK = einsum(self.Q, self.K, "... queries d_k, ... keys d_k -> ... queries keys")
        scores = QK / math.sqrt(self.Q.shape[-1])
        scores = scores.masked_fill(~self.mask, float("-inf"))
        scores = SoftMax().forward(scores, -1)
        return einsum(scores, self.V, "... queries keys, ... keys d_v -> ... queries d_v")