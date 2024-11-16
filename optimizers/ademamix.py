import torch
import math
from torch.optim.optimizer import Optimizer

class AdEMAMix(Optimizer):
    """
    Implements AdEMAMix optimizer, an extension of Adam with mixed EMA terms for enhanced convergence
    
    params (iterable) : Iterable of parameters to optimize or dictionaries defining parameter groups
    lr (float) : Learning rate (default: 1e-3)
    betas (Tuple[float, float, float]) : Coefficients for computing running averages of gradient and squared gradients (default: (0.9, 0.999, 0.9999))
    alpha (float) : Mixing parameter for adjusting influence of EMA terms (default: 5.0)
    T_beta3 (int) : Steps to reach final beta3 value (default: 200000)
    T_alpha (int) : Steps to reach final alpha value (default: 200000)
    eps (float) : Term added to denominator for numerical stability (default: 1e-8)
    weight_decay (float) : Weight decay coefficient (L2 regularization) (default: 0.0)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=5.0, T_beta3=200000, T_alpha=200000, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, alpha=alpha, T_beta3=T_beta3, T_alpha=T_alpha, eps=eps, weight_decay=weight_decay)
        super(AdEMAMix, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            T_beta3 = group["T_beta3"]
            T_alpha = group["T_alpha"]
            alpha_final = group["alpha"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdEMAMix ne supporte pas les gradients épars')

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["nu"] = torch.zeros_like(p)

                m1, m2, nu = state["m1"], state["m2"], state["nu"]
                state["step"] += 1
                step = state["step"]

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                alpha = self.alpha_scheduler(step, start=0, end=alpha_final, T=T_alpha)
                beta3 = self.beta3_scheduler(step, start=beta1, end=beta3_final, T=T_beta3)

                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                m2.mul_(beta3).add_(grad, alpha=1 - beta3)
                nu.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (nu.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update = (m1 / bias_correction1 + alpha * m2) / denom

                if lmbda != 0:
                    update.add_(p, alpha=lmbda)

                p.add_(-lr * update)
        return loss

    def alpha_scheduler(self, step, start, end, T):
        return min(step / T, 1.0) * (end - start) + start

    def beta3_scheduler(self, step, start, end, T):
        return min(step / T, 1.0) * (end - start) + start
