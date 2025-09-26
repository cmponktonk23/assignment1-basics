import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy_loss(
        inputs: Float[Tensor, " batch_size vocab_size"], 
        targets: Int[Tensor, " batch_size"]
        )->Float[Tensor, " batch_size"]:
    
    # Stablize input vector by subtract the maximum element
    stable: Float[Tensor, " batch_size vocab_size"] = inputs - inputs.max(dim=-1, keepdim=True).values
    
    # Align first dimension of stable and targets, use target value to index element in predicted logits
    numerator: Float[Tensor, " batch_size"] = stable[torch.arange(stable.size(0)), targets]

    # Element-wise sum exp(stable)
    denominator: Float[Tensor, " batch_size"] = stable.exp().sum(dim=-1, keepdim=True)
    
    # log(exp(numerator)) = numerator
    # numerator - log(denominator) = log(exp(numerator) / denominator)
    return -((numerator - denominator.log()).mean())
