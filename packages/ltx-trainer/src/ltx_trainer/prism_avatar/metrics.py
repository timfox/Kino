"""Marcel side-view artifact diagnostics (Eq. 14–15, Sec. 4.2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dilate_mask(mask: NDArray[np.floating], radius: int) -> NDArray[np.floating]:
    h, w = mask.shape
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ys = slice(max(0, dy), h + min(0, dy))
            xs = slice(max(0, dx), w + min(0, dx))
            ys_src = slice(max(0, -dy), h - max(0, dy))
            xs_src = slice(max(0, -dx), w - max(0, dx))
            out[ys, xs] = np.maximum(out[ys, xs], mask[ys_src, xs_src])
    return out


def neck_region(h: int, w: int) -> NDArray[np.bool_]:
    """Neck band N from Sec. 4.2."""
    ys = np.linspace(0, 1, h)[:, None]
    xs = np.linspace(0, 1, w)[None, :]
    return (ys >= 0.46) & (ys <= 0.90) & ((xs <= 0.42) | (xs >= 0.58))


def weighted_mean(values: NDArray[np.floating], weights: NDArray[np.floating]) -> float:
    w = weights.astype(np.float64)
    s = float(w.sum())
    if s <= 0:
        return 0.0
    return float((values * w).sum() / s)


def side_artifact_metrics(
    rgb: NDArray[np.floating],
    alpha: NDArray[np.floating],
    mesh: NDArray[np.floating],
    *,
    dilate_px: int = 9,
) -> dict[str, float]:
    """Eq. (14–15) diagnostics; lower is better except E_c handled separately."""
    a = alpha[..., 0] if alpha.ndim == 3 else alpha
    m = mesh[..., 0] if mesh.ndim == 3 else mesh
    d = dilate_mask(m, dilate_px)
    n = neck_region(a.shape[0], a.shape[1])
    alpha_sum = float(a.sum()) + 1e-8
    o_alpha = float((a * (1.0 - d)).sum() / alpha_sum)
    g_m = float((a[n] * (1.0 - d[n])).sum() / alpha_sum)
    g_d = float((a[n] * (1.0 - d[n])).sum() / max(n.sum(), 1))
    s_n = float((4.0 * a[n] * (1.0 - a[n])).sum() / max(n.sum(), 1))
    s_alpha = float((4.0 * a * (1.0 - a) * d).sum() / (d.sum() + 1e-8))
    b = (a > 0.03).astype(np.float64)
    if rgb.ndim == 3:
        c = rgb
        sat = c.max(axis=-1) - c.min(axis=-1)
    else:
        sat = np.zeros_like(a)
    ba = b * a
    f_c = (
        0.45 * weighted_mean(sat, ba)
        + 0.35 * weighted_mean(sat, a * (1.0 - d))
        + 0.20 * weighted_mean(sat, 4.0 * a * (1.0 - a) * d)
    )
    grad = np.abs(np.gradient(a, axis=0)).astype(np.float64) + np.abs(np.gradient(a, axis=1)).astype(np.float64)
    e_c = weighted_mean(grad, ba)
    return {
        "O_alpha": o_alpha,
        "G_m": g_m,
        "G_d": g_d,
        "S_n": s_n,
        "S_alpha": s_alpha,
        "F_c": f_c,
        "E_c": e_c,
    }
