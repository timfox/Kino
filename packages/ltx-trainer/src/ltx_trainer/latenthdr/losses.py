"""Losses for LatentHDR-style exposure supervision (``L_ev``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def exposure_latent_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted and target latents (same shape)."""
    return F.mse_loss(pred, target)
