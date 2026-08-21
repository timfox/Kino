"""StitchDiffusion stitch-block stub (Sec. 3.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def stitch_latents(latent: Tensor) -> Tensor:
    """Merge left/right ERP edges for seamless panorama."""
    w = latent.shape[-1]
    edge = max(1, w // 32)
    left = latent[..., :edge]
    right = latent[..., -edge:]
    blended = 0.5 * (left + right)
    out = latent.clone()
    out[..., :edge] = blended
    out[..., -edge:] = blended
    return out
