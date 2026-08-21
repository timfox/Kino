"""Self-consistent intrinsics from layer-0 pointmap (App. B)."""

from __future__ import annotations

import torch
from torch import Tensor


def fit_pinhole_intrinsics(
    points: Tensor,
    *,
    height: int,
    width: int,
    valid: Tensor | None = None,
) -> dict[str, float]:
    """Least-squares fit (fx, fy, cx, cy) from predicted camera-space XYZ."""
    if points.ndim == 5:
        points = points[:, 0]
    elif points.ndim == 4 and points.shape[0] > 3:
        points = points[0]
    z = points[..., 2].clamp_min(1e-4)
    xs = points[..., 0] / z
    ys = points[..., 1] / z
    h, w = points.shape[-3], points.shape[-2]
    v, u = torch.meshgrid(
        torch.arange(h, device=points.device, dtype=points.dtype),
        torch.arange(w, device=points.device, dtype=points.dtype),
        indexing="ij",
    )
    if valid is None:
        valid = z > 0
    mask = valid.reshape(-1)
    u_f = u.reshape(-1)[mask]
    v_f = v.reshape(-1)[mask]
    xs_f = xs.reshape(-1)[mask]
    ys_f = ys.reshape(-1)[mask]
    if u_f.numel() < 4:
        return {"fx": float(w), "fy": float(h), "cx": w / 2, "cy": h / 2}
    cx = u_f.mean()
    cy = v_f.mean()
    fx = ((u_f - cx) * xs_f).sum() / (xs_f**2).sum().clamp_min(1e-6)
    fy = ((v_f - cy) * ys_f).sum() / (ys_f**2).sum().clamp_min(1e-6)
    return {
        "fx": float(fx.item()),
        "fy": float(fy.item()),
        "cx": float(cx.item()),
        "cy": float(cy.item()),
    }
