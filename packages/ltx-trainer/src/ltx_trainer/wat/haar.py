"""One-level separable Haar DWT / IDWT for rank 1–3."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_SQRT2 = 2**0.5


def _haar_1d(x: Tensor) -> tuple[Tensor, Tensor]:
    """(B, C, T) with even T -> low/high (B, C, T//2)."""
    b, c, t = x.shape
    if t % 2:
        x = F.pad(x, (0, 1))
        t = x.shape[-1]
    x = x.reshape(b, c, t // 2, 2)
    lo = (x[..., 0] + x[..., 1]) / _SQRT2
    hi = (x[..., 0] - x[..., 1]) / _SQRT2
    return lo, hi


def _ihaar_1d(lo: Tensor, hi: Tensor) -> Tensor:
    b, c, n = lo.shape
    pairs = torch.stack([(lo + hi) / _SQRT2, (lo - hi) / _SQRT2], dim=-1)
    return pairs.reshape(b, c, n * 2)


def _dwt_along_axis(x: Tensor, axis: int) -> tuple[Tensor, Tensor]:
    perm = list(range(x.dim()))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    t = x.permute(*perm)
    lead = t.shape[:-1]
    length = t.shape[-1]
    flat = t.reshape(-1, length)
    # treat as (N, 1, L) batches
    lo, hi = _haar_1d(flat.unsqueeze(1))
    lo = lo.squeeze(1).reshape(*lead, lo.shape[-1])
    hi = hi.squeeze(1).reshape(*lead, hi.shape[-1])
    inv = [0] * x.dim()
    for i, p in enumerate(perm):
        inv[p] = i
    return lo.permute(*inv), hi.permute(*inv)


def _idwt_along_axis(lo: Tensor, hi: Tensor, axis: int) -> Tensor:
    perm = list(range(lo.dim()))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    lo_p = lo.permute(*perm)
    hi_p = hi.permute(*perm)
    lead = lo_p.shape[:-1]
    lo_f = lo_p.reshape(-1, lo_p.shape[-1])
    hi_f = hi_p.reshape(-1, hi_p.shape[-1])
    rec = _ihaar_1d(lo_f.unsqueeze(1), hi_f.unsqueeze(1)).squeeze(1)
    rec = rec.reshape(*lead, rec.shape[-1])
    inv = [0] * lo.dim()
    for i, p in enumerate(perm):
        inv[p] = i
    return rec.permute(*inv)


def haar_dwt(x: Tensor, rank: int) -> Tensor:
    """
    One-level separable Haar. Input:
      rank 1: (B, C, T)
      rank 2: (B, C, H, W)
      rank 3: (B, C, T, H, W)
    Output: (B, C * 2^rank, *downsampled_grid).
    """
    bands: list[Tensor] = [x]
    for axis in range(-rank, 0):
        nxt: list[Tensor] = []
        for b in bands:
            lo, hi = _dwt_along_axis(b, axis)
            nxt.extend([lo, hi])
        bands = nxt
    return torch.cat(bands, dim=1)


def haar_idwt(coeffs: Tensor, rank: int, channels: int) -> Tensor:
    """Inverse one-level Haar; coeffs (B, C*2^rank, *grid)."""
    n_sub = 2**rank
    parts = list(torch.chunk(coeffs, n_sub, dim=1))
    for axis in reversed(list(range(-rank, 0))):
        merged: list[Tensor] = []
        for i in range(0, len(parts), 2):
            merged.append(_idwt_along_axis(parts[i], parts[i + 1], axis))
        parts = merged
    return parts[0][:, :channels]
