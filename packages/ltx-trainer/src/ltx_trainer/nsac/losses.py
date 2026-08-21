"""Gaussian NLL and epistemic-separation regularizer (paper Eq. 7–10, Algorithm 1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def gaussian_nll(y: Tensor, mu: Tensor, log_std: Tensor) -> Tensor:
    """Scalar mean NLL for diagonal Gaussian (paper Eq. 7, Appendix B.1.2)."""
    inv_var = torch.exp(-2.0 * log_std)
    return 0.5 * ((math.log(2 * math.pi) + 2.0 * log_std + (y - mu).pow(2) * inv_var)).mean()


def epistemic_separation_loss(mu_stack_id: Tensor, mu_stack_ood: Tensor, eps: float = 1e-8) -> Tensor:
    """Eq. 9: ``log(1 + E[Var[μ_ID]] / (E[Var[μ_OOD]] + ε))``.

    ``mu_stack_*`` shape ``(N_mc, B, D)`` — variance across MC samples (stochastic attention).
    """
    v_id = mu_stack_id.var(dim=0, correction=0).mean()
    v_ood = mu_stack_ood.var(dim=0, correction=0).mean()
    return torch.log1p(v_id / (v_ood + eps))


class NSACTrainingLoss(nn.Module):
    """Wraps L_total = L_nll + λ L_reg when targets are available."""

    def __init__(self, n_mc: int, lambda_reg: float, mu_pert: float, sigma_pert: float, reg_eps: float = 1e-8) -> None:
        super().__init__()
        self.n_mc = n_mc
        self.lambda_reg = lambda_reg
        self.mu_pert = mu_pert
        self.sigma_pert = sigma_pert
        self.reg_eps = reg_eps

    def forward(self, forward_fn: nn.Module, x: Tensor, y: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """``forward_fn(x) -> (mu, log_std)`` each shape (B, D_out). Runs ``n_mc`` stochastic passes."""
        mus_id: list[Tensor] = []
        nlls: list[Tensor] = []
        for _ in range(self.n_mc):
            mu, log_std = forward_fn(x)
            mus_id.append(mu)
            nlls.append(gaussian_nll(y, mu, log_std))
        lnll = torch.stack(nlls, dim=0).mean()
        stack_id = torch.stack(mus_id, dim=0)

        xi = torch.randn_like(x) * self.sigma_pert + self.mu_pert
        x_ood = x + xi
        mus_ood: list[Tensor] = []
        for _ in range(self.n_mc):
            mu_o, _ = forward_fn(x_ood)
            mus_ood.append(mu_o)
        stack_ood = torch.stack(mus_ood, dim=0)
        lreg = epistemic_separation_loss(stack_id, stack_ood, self.reg_eps)
        total = lnll + self.lambda_reg * lreg
        return total, {"lnll": lnll.detach(), "lreg": lreg.detach(), "total": total.detach()}
