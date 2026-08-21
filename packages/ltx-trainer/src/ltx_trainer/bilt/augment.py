"""Spectral shift and noise augmentation (Sec. 2.2.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def spectral_shift(x: Tensor, shift: int) -> Tensor:
    """Sub-pixel spectral shift via linear interpolation on padded signal."""
    if shift == 0:
        return x
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(1)
        squeeze = True
    pad = max(abs(shift), 1)
    padded = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
    length = x.shape[-1]
    positions = torch.arange(length, device=x.device, dtype=torch.float32) + float(shift) + pad
    idx0 = positions.floor().long().clamp(0, padded.shape[-1] - 1)
    idx1 = (idx0 + 1).clamp(0, padded.shape[-1] - 1)
    w1 = positions - idx0.float()
    w0 = 1.0 - w1
    b, c, _ = padded.shape
    idx0_exp = idx0.view(1, 1, -1).expand(b, c, -1)
    idx1_exp = idx1.view(1, 1, -1).expand(b, c, -1)
    s0 = torch.gather(padded, -1, idx0_exp)
    s1 = torch.gather(padded, -1, idx1_exp)
    out = s0 * w0 + s1 * w1
    return out.squeeze(1) if squeeze else out


def apply_augmentation(
    x: Tensor,
    *,
    max_shift: int = 7,
    max_noise: float = 0.03,
    ratio: float = 0.65,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply random shift/noise to a fraction of batch samples."""
    out = x.clone()
    b = x.shape[0]
    for i in range(b):
        if torch.rand((), generator=generator).item() > ratio:
            continue
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=generator).item())
        xi = out[i]
        if xi.dim() == 1:
            xi = spectral_shift(xi.unsqueeze(0), shift).squeeze(0)
        else:
            xi = spectral_shift(xi.unsqueeze(0), shift).squeeze(0)
        noise_std = torch.rand((), generator=generator).item() * max_noise
        xi = xi + torch.randn_like(xi, generator=generator) * noise_std
        out[i] = xi.clamp(0.0, 1.0)
    return out
