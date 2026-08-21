"""Edit-region masks from synthetic source/target pairs (Sec. 4.3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def edit_region_mask(
    source: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.05,
) -> Tensor:
    """Binary mask M ∈ {0,1}^{H×W} from |I − Ĩ| (Sec. 4.3).

    Accepts (C,H,W) or (B,C,H,W); returns (H,W) or (B,H,W).
    """
    if source.shape != target.shape:
        raise ValueError("source and target must share shape")
    diff = (source - target).abs()
    if diff.dim() == 3:
        diff = diff.max(dim=0).values
    elif diff.dim() == 4:
        diff = diff.max(dim=1).values
    else:
        raise ValueError("expected (C,H,W) or (B,C,H,W)")
    return (diff > threshold).float()


def resize_mask_to_latent(mask: Tensor, latent_shape: tuple[int, ...]) -> Tensor:
    """Downsample image-space M to latent resolution."""
    h, w = latent_shape[-2], latent_shape[-1]
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:
        m = m.unsqueeze(1)
    return F.interpolate(m, size=(h, w), mode="nearest").squeeze(1)
