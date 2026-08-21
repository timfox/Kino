"""PBR path-tracer stub for R0/R1/O renders (Sec. 3.6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def pbr_render_pair(
    background: Tensor,
    object_mask: Tensor,
    hdr_env: Tensor,
    *,
    with_object: bool,
) -> Tensor:
    """
    Simulate receiver render R0 (no object) or R1 (with object).

    Uses HDR env mean intensity and mask occlusion as a cheap proxy.
    """
    env_gain = hdr_env.mean(dim=(-2, -1), keepdim=True).clamp(0.1, 4.0)
    lit = background * (0.6 + 0.4 * env_gain)
    if with_object:
        occ = F.max_pool2d(object_mask, 5, stride=1, padding=2)
        lit = lit * (1.0 - 0.55 * occ)
    return lit.clamp(0, 1)


def render_object_layer(
    background_shape: tuple[int, ...],
    object_rgb: Tensor,
    object_mask: Tensor,
    hdr_env: Tensor,
) -> Tensor:
    """Object O with HDR-tinted shading."""
    env_gain = hdr_env.mean(dim=(-2, -1), keepdim=True)
    shaded = object_rgb * (0.5 + 0.5 * env_gain)
    return shaded.clamp(0, 1)
