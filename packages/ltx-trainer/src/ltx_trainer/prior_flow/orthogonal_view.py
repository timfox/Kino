"""Orthogonal view generation (Sec. 3.1, Eq. 2–3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def primitive_to_orthogonal(frame: Tensor) -> Tensor:
    """
    T^o_p(Ip): 90° spherical x-rotation stub (Eq. 2–3).

    Production code reprojects via spherical rotation + bilinear sample; this stub
    rolls latitude to swap polar/equatorial emphasis while preserving [H, W].
    """
    return torch.roll(frame, shifts=frame.shape[2] // 4, dims=2)


def orthogonal_to_primitive(frame: Tensor) -> Tensor:
    """Inverse T^p_o for flow/confidence maps (Eq. 12)."""
    return torch.roll(frame, shifts=-(frame.shape[2] // 4), dims=2)


def flow_orthogonal_to_primitive(flow: Tensor) -> Tensor:
    """Fo2p = T^p_o(Fo): map orthogonal flow back to primitive coordinates."""
    return orthogonal_to_primitive(flow)


def horizontal_wrap_coords(x: Tensor, width: int) -> Tensor:
    """Modulo W for panoramic horizontal continuity (Eq. 4)."""
    return torch.remainder(x, float(width))
