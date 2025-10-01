import torch
from torch import Tensor
from jaxtyping import Int, Float
from .softmax import softmax
from .rms_norm import RMSNorm
from .embedding import Embedding
from .linear import Linear
from .transformer_block import TransformerBlock


class TransformerLM(torch.nn.Module):

    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        
        super().__init__()
        
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = torch.nn.ModuleList(
            TransformerBlock(
                d_model, 
                num_heads, 
                d_ff,
                context_length, 
                rope_theta,
                device=device,
                dtype=dtype) for i in range(num_layers))
        
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)


    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        # 1. token embedding
        in_features = self.token_embeddings.forward(in_indices)

        # 2. transformer blocks
        for transformer_block in self.layers:
            in_features = transformer_block.forward(in_features)
        
        # 3. norm
        out_features = self.ln_final.forward(in_features)
        
        # 4. linear (output embedding)
        out_features = self.lm_head.forward(out_features)

        return out_features # test case desired result has no softmax layer!!!
        
        # 5. softmax
        # return softmax(out_features, -1)
