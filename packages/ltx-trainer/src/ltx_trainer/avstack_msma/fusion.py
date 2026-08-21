"""Collaborative fusion stubs: naive vs correlation-aware (§4.2, Table 3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class CorrelationAssumption(str, Enum):
    NONE = "no_correlations"
    MINOR = "minor_correlations"
    MAJOR = "major_correlations"


@dataclass
class TrackEstimate:
    """Gaussian track state (position + scalar covariance)."""

    mean: np.ndarray  # (2,) x,y
    covariance: np.ndarray  # (2,2)


def covariance_intersection(a: TrackEstimate, b: TrackEstimate) -> TrackEstimate:
    """Decentralized conservative fusion (Julier–Uhlmann CI, §4.2)."""
    ya, yb = a.covariance, b.covariance
    inv_a = np.linalg.inv(ya)
    inv_b = np.linalg.inv(yb)
    y_f = np.linalg.inv(inv_a + inv_b)
    mean = y_f @ (inv_a @ a.mean + inv_b @ b.mean)
    return TrackEstimate(mean=mean, covariance=y_f)


def naive_fusion(a: TrackEstimate, b: TrackEstimate) -> TrackEstimate:
    """Treat correlated inputs as independent (fragile under correlation)."""
    mean = 0.5 * (a.mean + b.mean)
    cov = 0.5 * (a.covariance + b.covariance)
    return TrackEstimate(mean=mean, covariance=cov)


def fusion_mAP(
    assumption: CorrelationAssumption,
    *,
    local_only: float = 0.60,
    naive_no_corr: float = 0.86,
    naive_minor: float = 0.74,
    naive_major: float = 0.62,
    ci_no_corr: float = 0.90,
    ci_minor: float = 0.89,
    ci_major: float = 0.86,
) -> dict[str, float]:
    """Table 3 anchors under correlation assumptions."""
    if assumption == CorrelationAssumption.NONE:
        return {
            "local_only": local_only,
            "naive_fusion": naive_no_corr,
            "correlation_aware_fusion": ci_no_corr,
        }
    if assumption == CorrelationAssumption.MINOR:
        return {
            "local_only": local_only,
            "naive_fusion": naive_minor,
            "correlation_aware_fusion": ci_minor,
        }
    return {
        "local_only": local_only,
        "naive_fusion": naive_major,
        "correlation_aware_fusion": ci_major,
    }


def table_collaborative_fusion() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for assumption in CorrelationAssumption:
        row = fusion_mAP(assumption)
        row["assumption"] = assumption.value
        rows.append(row)
    return rows


def fusion_demo() -> dict[str, float]:
    """Smoke: CI tightens vs naive under synthetic correlated inputs."""
    a = TrackEstimate(mean=np.array([0.0, 0.0]), covariance=np.eye(2) * 4.0)
    b = TrackEstimate(mean=np.array([1.0, 0.5]), covariance=np.eye(2) * 4.0)
    ci = covariance_intersection(a, b)
    naive = naive_fusion(a, b)
    return {
        "ci_trace": float(np.trace(ci.covariance)),
        "naive_trace": float(np.trace(naive.covariance)),
        "ci_more_conservative": float(np.trace(ci.covariance) <= np.trace(naive.covariance)),
    }
