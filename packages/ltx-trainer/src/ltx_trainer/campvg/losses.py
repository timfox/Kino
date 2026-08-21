"""Diffusion training objective stub (Eq. 1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def diffusion_noise_loss(
    noise_pred: Tensor,
    noise_target: Tensor,
) -> Tensor:
    """Eq. (1) simplified MSE on noise prediction."""
    return F.mse_loss(noise_pred, noise_target)


def frame_reconstruction_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(pred, target)
