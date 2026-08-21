"""Motion Drift Loss and predictor objective (Eq. 5–7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lamo.latent_motion import macro_drift_vector


def schedule_weight(alpha_bar: Tensor) -> Tensor:
    """w(σ_t) := E_b[ᾱ_t,b]; damp drift at high noise (Sec. 3.2)."""
    return alpha_bar.mean()


def scale_normalized_drift_loss(
    mu_pred: Tensor,
    mu_target: Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Scale-normalized L2 drift loss (Eq. 5)."""
    num = (mu_pred - mu_target).pow(2).sum()
    denom = mu_target.detach().pow(2).sum() + epsilon
    return num / denom


def motion_drift_loss(
    x0_pred: Tensor,
    x0_target: Tensor,
    *,
    tau: int = 2,
    time_dim: int = 0,
    epsilon: float = 1e-6,
) -> Tensor:
    """L_drift between predicted and GT macro drifts on x̂0 pairs (Sec. 3.2)."""
    pred_delta = x0_pred.narrow(time_dim, tau, x0_pred.size(time_dim) - tau) - x0_pred.narrow(
        time_dim, 0, x0_pred.size(time_dim) - tau
    )
    tgt_delta = x0_target.narrow(time_dim, tau, x0_target.size(time_dim) - tau) - x0_target.narrow(
        time_dim, 0, x0_target.size(time_dim) - tau
    )
    n = pred_delta.size(time_dim)
    losses = []
    for i in range(n):
        if time_dim == 0:
            mu_hat = macro_drift_vector(pred_delta[i])
            mu_star = macro_drift_vector(tgt_delta[i])
        else:
            raise NotImplementedError("only time_dim=0 supported")
        losses.append(scale_normalized_drift_loss(mu_hat, mu_star, epsilon=epsilon))
    return torch.stack(losses).mean()


def predictor_loss(
    motion_field: Tensor,
    delta_target: Tensor,
    *,
    cosine_weight: float = 0.5,
) -> Tensor:
    """L_φ: MSE + α(1 − cos θ) (Eq. 7)."""
    mse = F.mse_loss(motion_field, delta_target)
    pred_flat = motion_field.reshape(motion_field.size(0), -1)
    tgt_flat = delta_target.reshape(delta_target.size(0), -1)
    cos = F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
    return mse + cosine_weight * (1.0 - cos)


def training_objective(
    l_denoise: Tensor,
    l_drift: Tensor,
    *,
    lambda_drift: float,
    alpha_bar: Tensor,
) -> Tensor:
    """L_train = L_denoise + λ_drift · w(σ_t) · L_drift (Eq. 6)."""
    w = schedule_weight(alpha_bar)
    return l_denoise + lambda_drift * w * l_drift
