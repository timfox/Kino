"""Unified navigation control and NavScore (WBENCH Sec. 3.1, Appendix C.4.1, Eq. 8–14)."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

Perspective = Literal["fpp", "tpp"]
NavKey = Literal["W", "S", "A", "D", "left", "right", "up", "down"]

# Table 5 — same key, different motion under FPP vs TPP
_FPP_TRANSLATION: dict[str, tuple[float, float, float]] = {
    "W": (0.0, 0.0, 1.0),
    "S": (0.0, 0.0, -1.0),
    "A": (-1.0, 0.0, 0.0),
    "D": (1.0, 0.0, 0.0),
}
_TPP_TRANSLATION: dict[str, tuple[float, float, float]] = {
    "W": (0.0, 0.0, 1.0),
    "S": (0.0, 0.0, -1.0),
    "A": (-1.0, 0.0, 0.0),
    "D": (1.0, 0.0, 0.0),
}
_ROTATION_YAW: dict[str, float] = {"left": 1.0, "right": -1.0}
_ROTATION_PITCH: dict[str, float] = {"up": 1.0, "down": -1.0}


def action_to_text(key: NavKey, perspective: Perspective = "fpp") -> str:
    """Natural-language navigation prompt for text-driven I2V models."""
    if key in _FPP_TRANSLATION:
        if perspective == "fpp":
            mapping = {
                "W": "The camera moves forward.",
                "S": "The camera moves backward.",
                "A": "The camera strafes left.",
                "D": "The camera strafes right.",
            }
        else:
            mapping = {
                "W": "The subject walks forward.",
                "S": "The subject steps backward.",
                "A": "The subject moves left.",
                "D": "The subject moves right.",
            }
        return mapping[key]
    rot = {
        "left": "turn left",
        "right": "turn right",
        "up": "tilt up",
        "down": "tilt down",
    }
    if perspective == "fpp":
        return f"The camera {rot[key]}."
    return f"The camera orbits {rot[key].replace('turn ', '')} around the subject."


def build_translation_trajectory(
    key: NavKey,
    length: float,
    num_points: int = 20,
) -> Tensor:
    """Straight-line GT in camera frame; length matched to predicted displacement."""
    if key not in _FPP_TRANSLATION:
        raise ValueError(f"not a translation key: {key}")
    direction = torch.tensor(_FPP_TRANSLATION[key], dtype=torch.float32)
    direction = direction / (direction.norm() + 1e-8)
    t = torch.linspace(0.0, length, num_points)
    return t.unsqueeze(-1) * direction.unsqueeze(0)


def build_orbit_trajectory(
    yaw_deg: float,
    radius: float,
    num_points: int = 20,
    *,
    tpp: bool = True,
) -> Tensor:
    """Adaptive orbital GT for rotation keys (Appendix C.4.1)."""
    if tpp and radius < 1.0:
        radius = 1.0
    theta = math.radians(yaw_deg)
    angles = torch.linspace(0.0, theta, num_points)
    x = radius * torch.sin(angles)
    z = radius * (1.0 - torch.cos(angles))
    y = torch.zeros_like(x)
    return torch.stack([x, y, z], dim=-1)


def arc_length_resample(positions: Tensor, num_points: int) -> Tensor:
    """Uniform resampling along cumulative arc length."""
    if positions.shape[0] <= 1:
        return positions
    seg = (positions[1:] - positions[:-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(1), seg.cumsum(0)])
    total = cum[-1].clamp_min(1e-8)
    targets = torch.linspace(0.0, total.item(), num_points)
    out = []
    for t_val in targets:
        idx = torch.searchsorted(cum, t_val).clamp(max=positions.shape[0] - 1)
        lo = max(int(idx.item()) - 1, 0)
        hi = int(idx.item())
        if hi == lo:
            out.append(positions[lo])
            continue
        alpha = (t_val - cum[lo]) / (cum[hi] - cum[lo] + 1e-8)
        out.append((1 - alpha) * positions[lo] + alpha * positions[hi])
    return torch.stack(out)


def absolute_trajectory_error(pred: Tensor, gt: Tensor) -> float:
    """Mean L2 after both trajectories are resampled to equal length."""
    k = min(pred.shape[0], gt.shape[0])
    if k == 0:
        return 0.0
    p = pred[:k]
    g = gt[:k]
    return float((p - g).norm(dim=-1).mean())


def normalized_ate_translation(ate: float, path_length: float, min_denom: float = 0.5) -> float:
    """Eq. (8): nATE_t."""
    denom = max(path_length, min_denom)
    return min(ate / denom, 1.0)


def normalized_ate_rotation(ate: float, total_rotation_deg: float, min_denom: float = 10.0) -> float:
    """Eq. (9): nATE_r."""
    denom = max(total_rotation_deg, min_denom)
    return min(ate / denom, 1.0)


def trajectory_consistency(nate_pairs: list[tuple[float, float]]) -> float:
    """Eq. (10)–(12): mean pairwise nATE over symmetric turn pairs; Cons = 1 - mean."""
    if not nate_pairs:
        return 1.0
    nt = sum(p[0] for p in nate_pairs) / len(nate_pairs)
    nr = sum(p[1] for p in nate_pairs) / len(nate_pairs)
    return 1.0 - (nt + nr) / 2.0


def nav_score(
    nate_t: float,
    nate_r: float,
    *,
    consistency: float | None = None,
    nate_pairs: list[tuple[float, float]] | None = None,
) -> float:
    """Eq. (13)–(14), score in [0, 1] before ×100 rescaling."""
    acc = 1.0 - (nate_t + nate_r) / 2.0
    cons = consistency if consistency is not None else trajectory_consistency(nate_pairs or [])
    return (acc + cons) / 2.0


def nav_score_from_trajectories(
    pred: Tensor,
    gt: Tensor,
    *,
    path_length: float | None = None,
    total_rotation_deg: float = 30.0,
    nate_pairs: list[tuple[float, float]] | None = None,
    num_samples: int = 20,
) -> float:
    """End-to-end NavScore smoke: resample, ATE, normalize, aggregate."""
    pred_r = arc_length_resample(pred, num_samples)
    gt_r = arc_length_resample(gt, num_samples)
    ate = absolute_trajectory_error(pred_r, gt_r)
    plen = path_length if path_length is not None else float(
        (pred_r[1:] - pred_r[:-1]).norm(dim=-1).sum()
    )
    nt = normalized_ate_translation(ate, plen)
    nr = normalized_ate_rotation(ate, total_rotation_deg)
    return nav_score(nt, nr, nate_pairs=nate_pairs)
