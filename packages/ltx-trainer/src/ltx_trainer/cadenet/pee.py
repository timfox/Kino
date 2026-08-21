"""Patch-Level Entropy Estimation — PEE (Sec. III-D, Eq. 1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def patch_entropy_reliability(gray: Tensor, *, patch: int = 16) -> Tensor:
    """Return reliability map ``R`` with values in [0,1], shape ``(H,W)``."""
    if gray.dim() == 3:
        gray = gray.mean(dim=0)
    h, w = gray.shape
    pad_h = (patch - h % patch) % patch
    pad_w = (patch - w % patch) % patch
    if pad_h or pad_w:
        gray = F.pad(gray.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze()
    h2, w2 = gray.shape
    patches = gray.unfold(0, patch, patch).unfold(1, patch, patch)
    ph, pw = patches.shape[0], patches.shape[1]
    flat = patches.reshape(ph * pw, -1)
    hist = torch.stack([(flat == b / 15.0).float().mean(dim=1) for b in range(16)], dim=1).clamp(min=1e-8)
    entropy = -(hist * torch.log2(hist)).sum(dim=1)
    r = 1.0 - entropy / 4.0  # log2(16)=4
    r_map = r.reshape(ph, pw)
    r_up = F.interpolate(r_map.unsqueeze(0).unsqueeze(0), size=(h2, w2), mode="bilinear", align_corners=False)
    return r_up.squeeze()[:h, :w].clamp(0.0, 1.0)


def reliability_at_box(r_map: Tensor, det_x1: float, det_y1: float, det_x2: float, det_y2: float) -> float:
    h, w = r_map.shape
    cx = int((det_x1 + det_x2) * 0.5)
    cy = int((det_y1 + det_y2) * 0.5)
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    return float(r_map[cy, cx])
