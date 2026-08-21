"""Cooling and facility thermals (Sec. 8)."""

from __future__ import annotations


def rack_requires_liquid_cooling(power_kw: float, *, threshold_kw: float = 35.0) -> bool:
    """Air cooling impractical above ~35 kW/rack (GB300 NVL72 ≈120 kW)."""
    return power_kw >= threshold_kw


def facility_it_power_with_pue(it_power_kw: float, pue: float) -> float:
    return float(it_power_kw * pue)


def energy_reuse_factor_waste_heat_mw(
    it_power_mw: float,
    pue: float,
    *,
    reuse_fraction: float = 0.4,
) -> float:
    """Recoverable thermal MW ≈ (PUE-1) * IT * reuse_fraction (ERF concept)."""
    waste = it_power_mw * max(pue - 1.0, 0.0)
    return float(waste * reuse_fraction)
