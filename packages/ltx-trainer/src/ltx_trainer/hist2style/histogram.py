"""Marginal YUV histogram + interactive slider edits (§3.3, §3.6)."""

from __future__ import annotations

import torch
from torch import Tensor


def marginal_histogram(image: Tensor, bins: int = 64) -> Tensor:
    """Per-channel normalized histogram; image B×3×H×W in [0,1]."""
    b, c, _, _ = image.shape
    hists = []
    for ch in range(c):
        flat = (image[:, ch].reshape(b, -1) * (bins - 1)).long().clamp(0, bins - 1)
        counts = torch.zeros(b, bins, device=image.device)
        for i in range(b):
            counts[i] = torch.bincount(flat[i], minlength=bins).float()
        hists.append(counts / (counts.sum(dim=-1, keepdim=True) + 1e-8))
    return torch.stack(hists, dim=1)  # B, 3, bins


def apply_sliders(
    hist: Tensor,
    *,
    exposure: float = 1.0,
    contrast: float = 1.0,
    u_shift: float = 0.0,
    v_shift: float = 0.0,
    smooth: float = 0.0,
) -> Tensor:
    """Y'CbCr-style slider edits on marginal histogram (§3.6)."""
    out = hist.clone()
    y = out[:, 0]
    y = y * exposure
    if contrast < 1.0:
        peak = y.argmax(dim=-1, keepdim=True)
        y = torch.zeros_like(y).scatter(-1, peak, 1.0) * (1 - contrast) + y * contrast
    out[:, 0] = y
    if u_shift != 0:
        out[:, 1] = torch.roll(out[:, 1], shifts=int(u_shift * 8), dims=-1)
    if v_shift != 0:
        out[:, 2] = torch.roll(out[:, 2], shifts=int(v_shift * 8), dims=-1)
    if smooth > 0:
        kernel = torch.tensor([0.25, 0.5, 0.25], device=hist.device)
        for ch in range(3):
            padded = torch.nn.functional.pad(out[:, ch : ch + 1], (1, 1), mode="replicate")
            sm = torch.nn.functional.conv1d(padded, kernel.view(1, 1, 3))
            out[:, ch] = (1 - smooth) * out[:, ch] + smooth * sm.squeeze(1)
    return out / (out.sum(dim=-1, keepdim=True) + 1e-8)
