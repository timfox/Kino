"""Subpixel prism encoding for glasses-free lenticular display (Eq. 11–13)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ltx_trainer.prism_avatar.config import DISPLAY_VIEW_MAX, DISPLAY_VIEW_MIN, PrismAvatarConfig


def subpixel_offset(channel: int, *, bgr: bool = False) -> int:
    """Map color channel index to subpixel offset ρ(c)."""
    order = [2, 1, 0] if bgr else [0, 1, 2]
    return order[channel % 3]


def view_row_offset(view_index: int, cfg: PrismAvatarConfig) -> float:
    """Eq. (11): s_i = s_ref − Δr/N (i − i_ref)."""
    n = cfg.num_views
    return cfg.display_s_ref - cfg.display_delta_r / n * (view_index - cfg.display_i_ref)


def candidate_subpixel_index(y: int, m: int, view_index: int, cfg: PrismAvatarConfig) -> float:
    """Eq. (12): b_i(y, m) = 3 C_r y + s_i + m Δr."""
    s_i = view_row_offset(view_index, cfg)
    return 3.0 * cfg.display_cr * y + s_i + m * cfg.display_delta_r


def routing_mask_for_view(
    height: int,
    width: int,
    view_index: int,
    cfg: PrismAvatarConfig,
    *,
    bgr: bool = False,
) -> NDArray[np.bool_]:
    """Binary mask M_i(p) for one virtual view (Eq. 13 setup)."""
    mask = np.zeros((height, width, 3), dtype=bool)
    n = cfg.num_views
    delta = cfg.display_delta_r
    max_m = int(np.ceil((3 * width + delta) / max(delta, 1e-6))) + 2
    for y in range(height):
        for x in range(width):
            for c in range(3):
                xs = 3 * x + subpixel_offset(c, bgr=bgr)
                for m in range(max_m):
                    b = candidate_subpixel_index(y, m, view_index, cfg)
                    b_floor = int(np.floor(b))
                    frac = b - b_floor
                    if b_floor == xs and frac < delta / n:
                        mask[y, x, c] = True
                        break
    return mask


def encode_panel_raster(
    views: list[NDArray[np.floating]],
    cfg: PrismAvatarConfig,
    *,
    bgr: bool = False,
) -> NDArray[np.floating]:
    """Eq. (13): E(p) = Σ_i M_i(p) V_i(u_p) — nearest-neighbor upsample stub."""
    h, w = cfg.panel_height, cfg.panel_width
    out = np.zeros((h, w, 3), dtype=np.float64)
    n_views = min(len(views), cfg.num_views)
    vh, vw = cfg.view_height, cfg.view_width
    for i in range(n_views):
        m = routing_mask_for_view(h, w, i, cfg, bgr=bgr)
        v = views[i]
        if v.shape[0] != vh or v.shape[1] != vw:
            vy = np.linspace(0, v.shape[0] - 1, h).astype(int)
            vx = np.linspace(0, v.shape[1] - 1, w).astype(int)
            v_up = v[vy][:, vx]
        else:
            vy = np.linspace(0, vh - 1, h).astype(int)
            vx = np.linspace(0, vw - 1, w).astype(int)
            v_up = v[vy][:, vx]
        out[m] = v_up[m]
    return np.clip(out, 0.0, 1.0)


def display_view_yaws(cfg: PrismAvatarConfig) -> NDArray[np.floating]:
    """Uniform yaw samples over [−25°, +25°] for N views."""
    return np.linspace(DISPLAY_VIEW_MIN, DISPLAY_VIEW_MAX, cfg.num_views)
