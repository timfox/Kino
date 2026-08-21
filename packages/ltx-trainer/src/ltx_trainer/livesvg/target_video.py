"""Synthetic target frame sequences for fitting smoke tests."""

from __future__ import annotations

import torch
from torch import Tensor


def synthetic_translating_disk(
    *,
    num_frames: int = 15,
    height: int = 64,
    width: int = 64,
    radius: int = 10,
    dx: float = 2.0,
    dy: float = 0.5,
    device: torch.device | None = None,
) -> Tensor:
    """``(T, 1, H, W)`` binary mask video — disk translates across the canvas."""
    device = device or torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    frames: list[Tensor] = []
    cx0 = width * 0.25
    cy0 = height * 0.5
    for t in range(num_frames):
        cx = cx0 + dx * t
        cy = cy0 + dy * t
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
        frames.append(disk.float().unsqueeze(0))
    return torch.stack(frames, dim=0)


def clean_target_background(target: Tensor, *, threshold: float = 0.5) -> Tensor:
    """Whitening outside a dilated foreground mask (suppl. H.0.3)."""
    if target.ndim != 4:
        raise ValueError("target must be (T, C, H, W)")
    out = target.clone()
    for t in range(out.shape[0]):
        mask = out[t, 0] > threshold
        # 1-pixel dilation via max pool
        m = mask.float().unsqueeze(0).unsqueeze(0)
        dilated = torch.nn.functional.max_pool2d(m, kernel_size=3, stride=1, padding=1)[0, 0] > 0.5
        out[t, :, ~dilated] = 1.0
    return out
