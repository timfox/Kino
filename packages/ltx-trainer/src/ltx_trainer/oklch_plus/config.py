"""Oklch+ config (arXiv:2606.05255)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OklchPlusParams:
    """Optimized COMBVD parameters (§3.4)."""

    alpha: float = 0.73
    n: float = 0.87
    sigma: float = 0.34


@dataclass(frozen=True)
class PowerLCParams:
    """Two-parameter Power-LC baseline (Table 2)."""

    alpha: float = 0.52
    gamma: float = 0.72


@dataclass(frozen=True)
class OklchPlusConfig:
    paper_arxiv: str = "arXiv:2606.05255"
    dataset: str = "COMBVD (3,813 pairs)"
    anchor_formula: str = "García et al. STRESS (Eqs. 1–2)"
    params: OklchPlusParams = OklchPlusParams()
    power_lc: PowerLCParams = PowerLCParams()
    packages: tuple[str, ...] = ("oklab", "transforms", "stress", "metrics", "simulation")
