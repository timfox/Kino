"""Level-1 ERP slicing into square views (Sec. 3.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def erp_boundary_stitch(frame: Tensor) -> Tensor:
    """Fourth slice: stitch left/right ERP boundaries (50% width each side)."""
    b, c, h, w = frame.shape
    quarter = w // 4
    left = frame[..., :, :quarter]
    right = frame[..., :, -quarter:]
    return torch.cat([right, left], dim=-1)


def erp_window_slices(frame: Tensor, *, window_ratio: float = 0.5) -> list[Tensor]:
    """Three overlapping square crops + boundary stitch from ERP [B,C,H,W]."""
    _, _, h, w = frame.shape
    win_w = max(h, w // 2)
    slices: list[Tensor] = []
    for start in (0, w // 4, w // 2):
        end = min(start + win_w, w)
        crop = frame[..., :, start:end]
        if crop.shape[-1] < h:
            crop = torch.nn.functional.pad(crop, (0, h - crop.shape[-1]))
        slices.append(crop[..., :h, :h])
    slices.append(erp_boundary_stitch(frame)[..., :h, :h])
    return slices
