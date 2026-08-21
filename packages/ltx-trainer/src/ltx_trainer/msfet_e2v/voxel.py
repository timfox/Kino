"""Event voxel grid representation (Eq. 3–4, Sec. III-B)."""

from __future__ import annotations

import torch
from torch import Tensor


def events_to_voxel(
    x: Tensor,
    y: Tensor,
    t: Tensor,
    p: Tensor,
    *,
    height: int,
    width: int,
    bins: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> Tensor:
    """Build V ∈ R^{H×W×B} from event tuples (x, y, t, p).

    Uses bilinear binning in normalized time (Eq. 3–4).
    """
    if not (x.shape == y.shape == t.shape == p.shape):
        raise ValueError("x, y, t, p must have the same shape")
    voxel = torch.zeros(bins, height, width, dtype=p.dtype, device=p.device)
    if x.numel() == 0:
        return voxel
    dt = max(t_end - t_start, 1e-8)
    t_star = (bins - 1) * (t - t_start) / dt
    xi = x.long().clamp(0, width - 1)
    yi = y.long().clamp(0, height - 1)
    b0 = torch.floor(t_star).long().clamp(0, bins - 1)
    b1 = (b0 + 1).clamp(0, bins - 1)
    w1 = (t_star - b0.float()).clamp(0.0, 1.0)
    w0 = 1.0 - w1
    for bi, w in ((b0, w0), (b1, w1)):
        voxel.index_put_((bi, yi, xi), p * w, accumulate=True)
    return voxel
