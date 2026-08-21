"""Visual-distractor single-task augmentation (Sec. 3.3, Table 3)."""

from __future__ import annotations

import random

import torch
from torch import Tensor


def draw_scribble_box(
    image: Tensor,
    *,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    box: tuple[int, int, int, int] | None = None,
    thickness: int = 2,
) -> Tensor:
    """Overlay a rectangular scribble on (C,H,W) in [0,1]."""
    out = image.clone()
    _, h, w = out.shape
    if box is None:
        bh, bw = h // 4, w // 4
        y0 = random.randint(0, max(0, h - bh - 1))
        x0 = random.randint(0, max(0, w - bw - 1))
        box = (y0, x0, y0 + bh, x0 + bw)
    y0, x0, y1, x1 = box
    c = torch.tensor(color, device=out.device, dtype=out.dtype).view(3, 1, 1)
    for t in range(thickness):
        yy0, yy1 = max(0, y0 - t), min(h, y1 + t)
        xx0, xx1 = max(0, x0 - t), min(w, x1 + t)
        if yy1 > yy0 and xx1 > xx0:
            stroke = c.expand(3, yy1 - yy0, xx1 - xx0)
            out[:, yy0:yy1, xx0:xx1] = stroke
    return out.clamp(0, 1)


def add_distractor_scribbles(
    scribbled: Tensor,
    *,
    count: int = 2,
    palette: list[tuple[float, float, float]] | None = None,
) -> Tensor:
    """Add 1–3 instruction-free scribbles (Sec. 3.3 falsification experiment)."""
    colors = palette or [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    out = scribbled
    for i in range(count):
        out = draw_scribble_box(out, color=colors[i % len(colors)])
    return out
