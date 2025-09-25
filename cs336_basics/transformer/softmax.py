import torch


def softmax(x: torch.Tensor, dim: int):
    # n * m * k => n * m * 1
    stable = x - x.max(dim=dim, keepdim=True).values
    # n * m * k
    numerator = stable.exp()
    # n * m * 1
    denominator = numerator.sum(dim=dim, keepdim=True)
    # n * m * k 
    return numerator / denominator
