import torch

class SoftMax(torch.nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, in_features: torch.Tensor, dim: int):
        # n * m * k => n * m * 1
        stable = in_features - in_features.max(dim=dim, keepdim=True).values
        # n * m * k
        numerator = stable.exp()
        # n * m * 1
        denominator = numerator.sum(dim=dim, keepdim=True)
        # n * m * k 
        return numerator / denominator