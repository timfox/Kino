"""Task-specific submetrics and IQR extraction (Sec. 4, Fig. 4)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from ltx_trainer.fbimmersion.config import ActivityName

# Skiing: CoG displacement + bilateral knee flexion (Fig. 3a–b)
SKIING_SUBMETRICS: tuple[str, ...] = (
    "cog_change",
    "right_knee_angle",
    "left_knee_angle",
)

# Boating standing: CoG sway (2D circle), trunk–leg angles (Fig. 3d–e)
BOATING_STANDING_SUBMETRICS: tuple[str, ...] = (
    "cog_sway_xy",
    "cog_chest_angle",
    "cog_feet_angle",
)

# Boating seated: CoP variation (2D circle), CoG–chest angle
BOATING_SEATED_SUBMETRICS: tuple[str, ...] = (
    "cop_xy",
    "cog_chest_angle",
    "cog_sway_xy",
)

SUBMETRICS_BY_ACTIVITY: dict[ActivityName, tuple[str, ...]] = {
    "skiing": SKIING_SUBMETRICS,
    "boating_standing": BOATING_STANDING_SUBMETRICS,
    "boating_seated": BOATING_SEATED_SUBMETRICS,
}

# 2D submetrics use minimum enclosing circle radius instead of IQR
SUBMETRIC_2D: frozenset[str] = frozenset({"cog_sway_xy", "cop_xy"})


def interquartile_range(values: Tensor | Sequence[float]) -> float:
    """IQR = Q3 − Q1 (Sec. 4)."""
    if not isinstance(values, Tensor):
        values = torch.tensor(list(values), dtype=torch.float64)
    if values.numel() == 0:
        return 0.0
    q1 = torch.quantile(values.float(), 0.25).item()
    q3 = torch.quantile(values.float(), 0.75).item()
    return float(q3 - q1)


def minimum_enclosing_circle_radius(points: Tensor) -> float:
    """Radius of minimum enclosing circle for 2D CoG/CoP variability (Sec. 4)."""
    if points.numel() == 0:
        return 0.0
    if points.dim() == 1:
        return float(points.std().item())
    if points.shape[-1] != 2:
        raise ValueError("points must be [N, 2]")
    center = points.mean(dim=0)
    radii = torch.norm(points - center, dim=-1)
    return float(radii.max().item())


def submetric_variability(name: str, series: Tensor) -> float:
    """Return IQR or 2D enclosing-circle radius for one submetric."""
    if name in SUBMETRIC_2D:
        if series.dim() == 1:
            # split interleaved x,y if needed
            n = series.numel() // 2
            pts = series[: n * 2].reshape(n, 2)
        else:
            pts = series
        return minimum_enclosing_circle_radius(pts)
    return interquartile_range(series)


def extract_submetric_values(
    activity: ActivityName,
    time_series: dict[str, Tensor],
) -> dict[str, float]:
    """Reduce time-series motion to scalar variability per submetric."""
    names = SUBMETRICS_BY_ACTIVITY[activity]
    out: dict[str, float] = {}
    for name in names:
        if name not in time_series:
            raise KeyError(f"missing submetric {name!r} for activity {activity!r}")
        out[name] = submetric_variability(name, time_series[name])
    return out


def radar_angles(n: int) -> Tensor:
    """θ_i = 2π(i−1)/n (Algorithm 1, line 4)."""
    i = torch.arange(n, dtype=torch.float64)
    return 2.0 * math.pi * i / n
