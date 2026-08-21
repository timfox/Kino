"""Retinex illumination prior (Sec. 3.2)."""

from __future__ import annotations

import torch
from torch import Tensor


def illumination_prior(image: Tensor) -> Tensor:
    """
    Channel-wise max illumination map ``Lp`` from RGB image ``[3,H,W]`` or ``[B,3,H,W]``.

    Returns ``[B,1,H,W]`` in ``[0,1]``.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    lp, _ = image.max(dim=1, keepdim=True)
    return lp.clamp(1e-6, 1.0)
