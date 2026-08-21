"""Training losses for H-OmniStereo (Eq. 3–5)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.h_omnistereo.heading_normal import angular_loss_deg


def smooth_l1(x: Tensor, y: Tensor, beta: float = 1.0) -> Tensor:
    return torch.nn.functional.smooth_l1_loss(x, y, beta=beta, reduction="mean")


def disparity_loss(
    predictions: list[Tensor],
    target: Tensor,
    *,
    gamma: float = 0.9,
) -> Tensor:
    """L_disp: smooth L1 on d0 + γ-weighted L1 on refinements (Eq. 4)."""
    if not predictions:
        raise ValueError("predictions must be non-empty")
    loss = smooth_l1(predictions[0], target)
    k_total = len(predictions)
    for k, dk in enumerate(predictions[1:], start=1):
        w = gamma ** (k_total - k)
        loss = loss + w * torch.nn.functional.l1_loss(dk, target)
    return loss


def uncertainty_nll_loss(
    disparities: list[Tensor],
    sigmas: list[Tensor],
    target: Tensor,
    *,
    gamma: float = 0.9,
) -> Tensor:
    """L_conf negative log-likelihood (Eq. 5); σ = exp(u)."""
    loss = torch.zeros((), device=target.device, dtype=target.dtype)
    k_total = len(sigmas)
    for k, (dk, sigma) in enumerate(zip(disparities, sigmas, strict=True)):
        w = gamma ** (k_total - k)
        err = torch.abs(dk - target)
        term = (err / sigma.clamp(min=1e-4) + torch.log(sigma.clamp(min=1e-4))).mean()
        loss = loss + w * term
    return loss / max(k_total, 1)


def normal_training_loss(pred: Tensor, target: Tensor) -> Tensor:
    return angular_loss_deg(pred, target)
