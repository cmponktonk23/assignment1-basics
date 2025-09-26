import torch
import math
from typing import Callable, Optional

class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr, betas, eps, weight_decay):

        defaults = {
            'lr': lr,
            'beta1': betas[0],
            'beta2': betas[1],
            'eps': eps,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)



    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            alpha, beta1, beta2, eps, weight_decay = group["lr"], group["beta1"], group["beta2"], group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                g = p.grad

                m = m * beta1 + (1 - beta1) * g
                v = v * beta2 + (1 - beta2) * g * g
                alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= alpha_t * m / (v.sqrt() + eps)
                p.data -= alpha * weight_decay * p.data

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss

