"""Runtime and Bjøntegaard-style metrics (Sec. 4.3, Eq. 8–9)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


def delta_t_serial(times_method: Sequence[float], times_ref: Sequence[float]) -> float:
    """ΔT_S (Eq. 8): relative total serial encode time change."""
    ref = sum(times_ref)
    if ref <= 0:
        return 0.0
    return (sum(times_method) - ref) / ref


def delta_t_parallel(times_method: Sequence[float], times_ref: Sequence[float]) -> float:
    """ΔT_P (Eq. 9): relative change vs slowest representation."""
    ref_max = max(times_ref)
    if ref_max <= 0:
        return 0.0
    return (max(times_method) - ref_max) / ref_max


def bdet_stub(
    time_curve_method: Tensor,
    time_curve_ref: Tensor,
    *,
    quality_curve_method: Tensor | None = None,
    quality_curve_ref: Tensor | None = None,
) -> float:
    """BDET proxy: log-average time ratio when quality curves match (negative = faster)."""
    if time_curve_method.numel() < 2:
        return 0.0
    ratio = (time_curve_method / time_curve_ref.clamp_min(1e-6)).log().mean()
    return float(ratio.item() * 100.0)


def speedup_factor(time_method: float, time_ref: float) -> float:
    if time_method <= 0:
        return 1.0
    return time_ref / time_method
