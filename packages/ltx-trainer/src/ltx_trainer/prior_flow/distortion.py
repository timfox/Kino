"""Primitive vs orthogonal distortion maps (Fig. 3)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def latitude_distortion(height: int, device: torch.device | None = None) -> Tensor:
    """|cos(latitude)| — high at equator, low at poles (ERP sampling)."""
    v = torch.arange(height, device=device, dtype=torch.float32)
    phi = math.pi / 2.0 - (v + 0.5) * (math.pi / height)
    return torch.cos(phi).abs()


def primitive_distortion_map(height: int, width: int, device: torch.device | None = None) -> Tensor:
    """Bright near equator, dark at poles."""
    row = latitude_distortion(height, device)
    return row.view(1, 1, height, 1).expand(1, 1, height, width)


def orthogonal_distortion_map(height: int, width: int, device: torch.device | None = None) -> Tensor:
    """Opposite pattern: poles low-distortion in orthogonal view."""
    prim = primitive_distortion_map(height, width, device)
    return 1.0 - prim + prim.min()
