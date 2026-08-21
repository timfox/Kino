"""Sinusoidal spherical encoding γ(λ, φ) (Eq. 7)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sinusoidal_spherical_encoding(lam: Tensor, phi: Tensor, num_bands: int = 8) -> Tensor:
    """
    Fixed γ(λ, φ) with sin/cos at multiple frequencies.

    Returns [..., 4 * num_bands].
    """
    freqs = 2.0 ** torch.arange(num_bands, device=lam.device, dtype=lam.dtype) * math.pi
    feats: list[Tensor] = []
    for f in freqs:
        feats.extend(
            [
                torch.sin(f * lam),
                torch.cos(f * lam),
                torch.sin(f * phi),
                torch.cos(f * phi),
            ]
        )
    return torch.stack(feats, dim=-1)
