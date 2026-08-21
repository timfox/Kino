"""Equirectangular projection and vertical shift (Sec. 3.1–3.2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def vertical_shift_rows(erp: Tensor, pitch_deg: float, erp_h: int) -> Tensor:
    """
    Shift ERP rows so perspective horizon aligns with target latitude (Fig. 2).
    Positive pitch moves content upward in ERP (negative row shift).
    """
    shift = int(round(-pitch_deg / 180.0 * erp_h / 2.0))
    return torch.roll(erp, shifts=shift, dims=-2)


def crop_informative_columns(erp: Tensor, target_w: int) -> Tensor:
    """Keep center informative columns for narrow-FoV perspective (Sec. 3.3.1)."""
    _, _, h, w = erp.shape
    if w <= target_w:
        return erp
    start = (w - target_w) // 2
    return erp[..., :, start : start + target_w]


def pad_to_pano_width(pp_feat: Tensor, pano_w: int) -> Tensor:
    """Zero-pad perspective feature map width before transformer concat."""
    _, c, h, w = pp_feat.shape
    if w >= pano_w:
        return pp_feat
    pad = pano_w - w
    left = pad // 2
    right = pad - left
    return torch.nn.functional.pad(pp_feat, (left, right, 0, 0))
