"""Frequency-Focused Supervision (Sec. 4.2, Eq. 5–8)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def flow_matching_loss(
    velocity_pred: Tensor,
    target_velocity: Tensor,
) -> Tensor:
    """L_FM = ||ν(z_t,c,t) − (ϵ − y)||^2 (Eq. 5)."""
    return F.mse_loss(velocity_pred, target_velocity)


def dft2(x: Tensor) -> Tensor:
    """Orthonormal 2D DFT magnitude spectrum helper."""
    if x.dim() == 2:
        return torch.fft.fft2(x, norm="ortho")
    if x.dim() == 4:
        return torch.fft.fft2(x, norm="ortho")
    raise ValueError("expected (H,W) or (B,C,H,W) tensor")


def frequency_weight(
    delta_f: Tensor,
    *,
    alpha_t: float,
    epsilon: float = 1e-8,
) -> Tensor:
    """W(ΔF, α_t) = (|ΔF|+ε)^α / max(|ΔF|+ε)^α (Eq. 7)."""
    mag = (delta_f.abs() + epsilon).pow(alpha_t)
    return mag / mag.max().clamp(min=epsilon)


def focus_intensity(
    t: Tensor | float,
    *,
    alpha_min: float = 0.2,
    alpha_max: float = 1.2,
    gamma: float = 2.0,
) -> Tensor | float:
    """α_t = α_min + (α_max − α_min)(1 − t)^γ (Sec. 4.2)."""
    if isinstance(t, Tensor):
        return alpha_min + (alpha_max - alpha_min) * (1.0 - t).pow(gamma)
    return alpha_min + (alpha_max - alpha_min) * ((1.0 - float(t)) ** gamma)


def frequency_focused_loss(
    y_hat: Tensor,
    y: Tensor,
    *,
    t: float = 0.5,
    alpha_min: float = 0.2,
    alpha_max: float = 1.2,
    gamma: float = 2.0,
    epsilon: float = 1e-8,
) -> Tensor:
    """L_freq (Eq. 8): weighted spectral discrepancy."""
    if y_hat.dim() == 4:
        y_hat = y_hat.mean(dim=1)
        y = y.mean(dim=1)
    if y_hat.dim() == 3:
        y_hat = y_hat[0]
        y = y[0]
    delta = dft2(y_hat) - dft2(y)
    alpha_t = focus_intensity(t, alpha_min=alpha_min, alpha_max=alpha_max, gamma=gamma)
    w = frequency_weight(delta, alpha_t=float(alpha_t), epsilon=epsilon)
    return (w * delta.abs()).mean()


def combined_training_loss(
    velocity_pred: Tensor,
    target_velocity: Tensor,
    y_hat: Tensor,
    y: Tensor,
    *,
    t: float = 0.5,
    lambda_freq: float = 1.0,
    alpha_min: float = 0.2,
    alpha_max: float = 1.2,
    gamma: float = 2.0,
) -> dict[str, Tensor]:
    """L = L_FM + λ L_freq (Sec. 4.2)."""
    l_fm = flow_matching_loss(velocity_pred, target_velocity)
    l_freq = frequency_focused_loss(
        y_hat,
        y,
        t=t,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        gamma=gamma,
    )
    return {
        "loss": l_fm + lambda_freq * l_freq,
        "loss_fm": l_fm.detach(),
        "loss_freq": l_freq.detach(),
    }
