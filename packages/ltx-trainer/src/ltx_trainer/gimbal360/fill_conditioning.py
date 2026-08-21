"""Flux-fill style channel conditioning (Sec. 3.1, Eq. z_t ⊕ M ⊕ z_ref)."""

from __future__ import annotations

import torch
from torch import Tensor


def concat_fill_channels(z_t: Tensor, mask: Tensor, z_ref: Tensor) -> Tensor:
    """Concatenate noisy latent, binary mask, and reference latent along channels."""
    if mask.shape[-2:] != z_t.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask, size=z_t.shape[-2:], mode="nearest")
    if z_ref.shape[-2:] != z_t.shape[-2:]:
        z_ref = torch.nn.functional.interpolate(z_ref, size=z_t.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([z_t, mask, z_ref], dim=1)


def fill_channel_count(latent_channels: int) -> int:
    return latent_channels * 2 + 1
