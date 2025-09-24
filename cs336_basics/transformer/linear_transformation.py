import math
import torch


class Linear(torch.nn.Module):

    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.W = torch.nn.Parameter(torch.empty(in_features, out_features))

        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.W, 
            mean = 0, 
            std = std,
            a = -3 * std,
            b = 3 * std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W
