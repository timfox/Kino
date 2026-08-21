"""Pseudo-multiview side-frame selection, ranking, and alignment (Sec. 3.2–3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ltx_trainer.prism_avatar.config import (
    YAW_ACCEPT_MAX,
    YAW_ACCEPT_MIN,
    YAW_BIN_TOLERANCE,
    PrismAvatarConfig,
)


@dataclass
class PMVFrame:
    index: int
    yaw: float
    bin_yaw: float
    score: float
    align_ok: bool


def yaw_bin_and_accept(yaw_deg: float, cfg: PrismAvatarConfig) -> tuple[float, float, int]:
    """Eq. (3–4): snap yaw to nearest lateral bin and accept flag."""
    bins = np.asarray(cfg.yaw_bins, dtype=np.float64)
    b = float(bins[np.argmin(np.abs(np.abs(yaw_deg) - bins))])
    a = int(
        YAW_ACCEPT_MIN <= abs(yaw_deg) <= YAW_ACCEPT_MAX
        and abs(abs(yaw_deg) - b) <= YAW_BIN_TOLERANCE
    )
    y_hat = float(np.sign(yaw_deg) * b) if yaw_deg != 0 else 0.0
    return y_hat, b, a


def side_frame_score(
    yaw_deg: float,
    bin_yaw: float,
    *,
    matte_clean: float,
    expr_stable: float,
    lateral_support: float,
    boundary_risk: float,
    collar_risk: float,
) -> float:
    """Eq. (5) simplified scalar ranking."""
    w_yaw = 1.0 - min(abs(abs(yaw_deg) - bin_yaw) / 2.5, 1.0)
    positive = 0.35 * w_yaw + 0.25 * matte_clean + 0.20 * expr_stable + 0.20 * lateral_support
    risk = 0.30 * boundary_risk + 0.20 * collar_risk
    return float(positive - risk)


def select_pmv_frames(
    yaws: NDArray[np.floating],
    scores: NDArray[np.floating] | None,
    cfg: PrismAvatarConfig,
    *,
    per_bin_cap: int = 3,
) -> list[PMVFrame]:
    """Keep high-scoring yaw-valid frames with balanced left/right bins."""
    if scores is None:
        scores = np.ones_like(yaws, dtype=np.float64)
    selected: list[PMVFrame] = []
    for side in (-1.0, 1.0):
        for b in cfg.yaw_bins:
            candidates: list[PMVFrame] = []
            for i, yaw in enumerate(yaws):
                y_hat, bin_y, acc = yaw_bin_and_accept(float(yaw), cfg)
                if not acc or np.sign(yaw) != side or abs(bin_y - b) > 1e-6:
                    continue
                candidates.append(
                    PMVFrame(index=i, yaw=float(yaw), bin_yaw=y_hat, score=float(scores[i]), align_ok=True)
                )
            candidates.sort(key=lambda f: f.score, reverse=True)
            selected.extend(candidates[:per_bin_cap])
    return selected


def alignment_gate(
    iou: float,
    iou_edge: float,
    centroid_dist_px: float,
    cfg: PrismAvatarConfig,
) -> bool:
    """Eq. (7) alignment acceptance."""
    return (
        iou >= cfg.align_iou_min
        and iou_edge >= cfg.align_iou_edge_min
        and centroid_dist_px <= cfg.align_centroid_max_px
    )


def virtual_camera(yaw_deg: float, cfg: PrismAvatarConfig) -> dict[str, float]:
    """Eq. (6) horizontal arc camera around avatar center."""
    theta = np.deg2rad(yaw_deg)
    r = cfg.camera_radius
    return {
        "yaw_deg": float(yaw_deg),
        "radius": r,
        "origin_x": float(r * np.sin(theta)),
        "origin_z": float(r * np.cos(theta)),
        "look_at_y_offset": 0.08,
    }
