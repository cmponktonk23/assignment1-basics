from torch import Tensor
from jaxtyping import Int
from .softmax import SoftMax
from .rms_norm import RMSNorm
from .embedding import Embedding
from .linear_transformation import Linear
from .transformer_block import TransformerBlock


class TransformLM:

    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 weights: dict[str, Tensor]):
        
        self.token_embedding = Embedding(vocab_size, d_model)
        self.token_embedding.load_state_dict({ 'embedding_matrix': weights['token_embeddings.weight'] })

        self.transformer_blocks = [TransformerBlock(
            d_model, 
            num_heads, 
            d_ff,
            context_length, 
            rope_theta,
            {
                'attn.q_proj.weight': weights[f"layers.{i}.attn.q_proj.weight"],
                'attn.k_proj.weight': weights[f"layers.{i}.attn.k_proj.weight"],
                'attn.v_proj.weight': weights[f"layers.{i}.attn.v_proj.weight"],
                'attn.output_proj.weight': weights[f"layers.{i}.attn.output_proj.weight"],
                'ffn.w1.weight': weights[f"layers.{i}.ffn.w1.weight"],
                'ffn.w2.weight': weights[f"layers.{i}.ffn.w2.weight"],
                'ffn.w3.weight': weights[f"layers.{i}.ffn.w3.weight"],
                'ln1.weight': weights[f"layers.{i}.ln1.weight"],
                'ln2.weight': weights[f"layers.{i}.ln2.weight"],
            },
            ) for i in range(num_layers)]
        
        self.final_rmsnorm = RMSNorm(d_model)
        self.final_rmsnorm.load_state_dict({ 'g': weights['ln_final.weight'] })

        self.linear = Linear(d_model, vocab_size)
        self.linear.load_state_dict({ 'W': weights['lm_head.weight'].t() })

        self.softmax = SoftMax()


    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):
        # 1. token embedding
        in_features = self.token_embedding.forward(in_indices)

        # 2. transformer blocks
        for transformer_block in self.transformer_blocks:
            in_features = transformer_block.forward(in_features)
        
        # 3. norm
        out_features = self.final_rmsnorm.forward(in_features)
        
        # 4. linear (output embedding)
        out_features = self.linear.forward(out_features)

        return out_features # test case desired result has no softmax layer!!!
        
        # 5. softmax
        # return self.softmax.forward(out_features, -1)
