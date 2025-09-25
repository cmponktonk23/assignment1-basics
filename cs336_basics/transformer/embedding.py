import torch


class Embedding(torch.nn.Module):

    def __init__(self, 
                 num_embeddings: int,                 # vocab_size
                 embedding_dim: int,                  # d_model
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        
        super().__init__()

        self.embedding_matrix = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        torch.nn.init.trunc_normal_(
            self.embedding_matrix, 
            mean = 0,
            std = 1,
            a = -3,
            b = 3)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]
