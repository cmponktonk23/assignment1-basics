import torch

class SwiGLU(torch.nn.Module):

    def __init__(self, 
                 d_model: int, 
                 d_ff: int,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = torch.nn.Parameter(torch.empty(d_model, d_ff))
        self.W2 = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.W3 = torch.nn.Parameter(torch.empty(d_model, d_ff))


    @classmethod
    def silu(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


    def forward(self, x: torch.Tensor):
        return (SwiGLU.silu(x @ self.W1) * (x @ self.W3)) @ self.W2
