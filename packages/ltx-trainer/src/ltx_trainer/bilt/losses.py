"""BiLT loss functions (Sec. 2.2.2, Eq. 1–4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.bilt.config import (
    LATENT_EXCL_LAMBDA,
    LOG_EPS,
    LOG_LOSS_ALPHA,
    MU_A_WEIGHT,
    SWAP_LOSS_BETA,
)


def log_scale_loss(mu_a_pred: Tensor, mu_a_true: Tensor, *, eps: float = LOG_EPS) -> Tensor:
    """L_log (Eq. 2)."""
    return F.mse_loss(torch.log(mu_a_pred + eps), torch.log(mu_a_true + eps))


def swap_penalty(mu_a_pred: Tensor, mu_a_true: Tensor, mu_s_pred: Tensor, mu_s_true: Tensor) -> Tensor:
    """L_swap (Eq. 3)."""
    ma_p = mu_a_pred.mean(dim=-1)
    ma_t = mu_a_true.mean(dim=-1)
    ms_p = mu_s_pred.mean(dim=-1)
    ms_t = mu_s_true.mean(dim=-1)
    return (F.relu(ma_p - ma_t) * F.relu(ms_t - ms_p)).mean()


def latent_loss(z: Tensor, *, lambda_excl: float = LATENT_EXCL_LAMBDA) -> Tensor:
    """L_latent = L_neg + λ_s L_excl (Eq. 4)."""
    l_neg = F.relu(-z).mean()
    l_excl = (z[:, 1] * z[:, 2]).mean() if z.shape[1] >= 3 else torch.tensor(0.0, device=z.device)
    return l_neg + lambda_excl * l_excl


def reconstruction_loss(
    mu_a_pred: Tensor,
    mu_a_true: Tensor,
    mu_s_pred: Tensor,
    mu_s_true: Tensor,
) -> Tensor:
    """L_output (Eq. 1)."""
    mse_a = F.mse_loss(mu_a_pred, mu_a_true)
    mse_s = F.mse_loss(mu_s_pred, mu_s_true)
    l_log = log_scale_loss(mu_a_pred, mu_a_true)
    l_swap = swap_penalty(mu_a_pred, mu_a_true, mu_s_pred, mu_s_true)
    return MU_A_WEIGHT * mse_a + mse_s + LOG_LOSS_ALPHA * l_log + SWAP_LOSS_BETA * l_swap
