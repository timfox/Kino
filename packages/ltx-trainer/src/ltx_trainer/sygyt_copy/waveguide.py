"""Kelly–Lochbaum waveguide stubs (Sec. 3.2)."""

from __future__ import annotations


def reflection_coefficient(area_left: float, area_right: float) -> float:
    denom = area_left + area_right
    if denom == 0.0:
        return 0.0
    return (area_left - area_right) / denom


def apply_damping(value: float, damping: float) -> float:
    return value * damping


def clip_damping(d: float, lo: float = 0.99, hi: float = 0.9999) -> float:
    return max(lo, min(hi, d))


def three_way_scattering(area_l: float, area_r: float, area_b: float) -> tuple[float, float, float]:
    sigma = area_l + area_r + area_b
    if sigma == 0.0:
        return 0.0, 0.0, 0.0
    return (
        (2.0 * area_l - sigma) / sigma,
        (2.0 * area_r - sigma) / sigma,
        (2.0 * area_b - sigma) / sigma,
    )
