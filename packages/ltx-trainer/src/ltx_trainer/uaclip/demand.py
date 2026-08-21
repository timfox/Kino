"""Platform demand models and visual utility h_v(v) (Sections 5.2–5.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Platform = Literal["amazon", "airbnb"]


@dataclass(frozen=True)
class AmazonVisualCoeffs:
    """Table 1 — log(sales rank); lower prediction ⇒ higher demand."""

    colorfulness_lin: float = -2.249
    colorfulness_quad: float = 2.118
    brightness_lin: float = -0.281
    brightness_quad: float = 0.588
    symmetry_lin: float = -1.074
    symmetry_quad: float = 1.664
    aesthetic_lin: float = -2.885
    aesthetic_quad: float = 2.951


@dataclass(frozen=True)
class AirbnbVisualCoeffs:
    """Table 3 — log(occupancy rate); higher prediction ⇒ higher demand."""

    uniqueness_lin: float = 0.288
    uniqueness_quad: float = -0.241
    aesthetic_lin: float = 0.978
    aesthetic_quad: float = -1.013


def _quad_util(x: float, lin: float, quad: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return lin * x + quad * (x * x)


def amazon_log_sales_rank(
    attrs: dict[str, float],
    coeffs: AmazonVisualCoeffs | None = None,
) -> float:
    """Predicted log sales rank from rescaled visual attributes in [0,1]."""
    c = coeffs or AmazonVisualCoeffs()
    return (
        _quad_util(attrs.get("colorfulness", 0.5), c.colorfulness_lin, c.colorfulness_quad)
        + _quad_util(attrs.get("brightness", 0.5), c.brightness_lin, c.brightness_quad)
        + _quad_util(attrs.get("symmetry", 0.5), c.symmetry_lin, c.symmetry_quad)
        + _quad_util(attrs.get("aesthetic", 0.5), c.aesthetic_lin, c.aesthetic_quad)
    )


def amazon_demand_score(attrs: dict[str, float], coeffs: AmazonVisualCoeffs | None = None) -> float:
    """Higher is better (inverse of log rank proxy)."""
    return -amazon_log_sales_rank(attrs, coeffs)


def airbnb_log_occupancy(
    attrs: dict[str, float],
    coeffs: AirbnbVisualCoeffs | None = None,
) -> float:
    """Visual component of log occupancy (Table 3 visual terms only)."""
    c = coeffs or AirbnbVisualCoeffs()
    return _quad_util(attrs.get("uniqueness", 0.5), c.uniqueness_lin, c.uniqueness_quad) + _quad_util(
        attrs.get("aesthetic", 0.5), c.aesthetic_lin, c.aesthetic_quad
    )


def airbnb_demand_score(attrs: dict[str, float], coeffs: AirbnbVisualCoeffs | None = None) -> float:
    return airbnb_log_occupancy(attrs, coeffs)


def visual_utility(
    attrs: dict[str, float],
    platform: Platform,
    *,
    eta: float = 0.5,
) -> float:
    """h_v(v) used in Utility-Aware CLIP score (Eq. 3 / Section 5.2.2)."""
    if platform == "amazon":
        return eta * amazon_log_sales_rank(attrs)
    return eta * airbnb_log_occupancy(attrs)


def utility_regularized_similarity(
    clip_sim: float,
    attrs: dict[str, float],
    platform: Platform,
    *,
    eta: float = 0.5,
) -> float:
    """U S(v,t) = η·demand_visual(attrs) + s_θ(v,t) (Section 5.2.2)."""
    return visual_utility(attrs, platform, eta=eta) + clip_sim


def inverted_u_peak(lin: float, quad: float) -> float:
    """Argmax of lin·x + quad·x² on [0,1] for quad < 0 (Airbnb uniqueness)."""
    if abs(quad) < 1e-12:
        return 1.0 if lin > 0 else 0.0
    x = -lin / (2.0 * quad)
    return float(np.clip(x, 0.0, 1.0))


def amazon_optimal_attrs(coeffs: AmazonVisualCoeffs | None = None) -> dict[str, float]:
    """Per-attribute peaks under inverted-U (quad > 0 on log rank ⇒ peak in (0,1))."""
    c = coeffs or AmazonVisualCoeffs()
    return {
        "colorfulness": inverted_u_peak(-c.colorfulness_lin, c.colorfulness_quad),
        "brightness": inverted_u_peak(-c.brightness_lin, c.brightness_quad),
        "symmetry": inverted_u_peak(-c.symmetry_lin, c.symmetry_quad),
        "aesthetic": inverted_u_peak(-c.aesthetic_lin, c.aesthetic_quad),
    }
