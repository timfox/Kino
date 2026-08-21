"""Paper tables and supplementary data for sparse 4D BCV (arXiv:2605.19160)."""

from __future__ import annotations

from typing import Any


def table1_metrics_summary() -> list[dict[str, str]]:
    """Table 1 — six 4D evaluation metrics (paper taxonomy)."""
    return [
        {"metric": "MSE", "category": "Global error & fidelity", "characteristics": "L2; penalizes outliers"},
        {"metric": "PSNR", "category": "Global error & fidelity", "characteristics": "Log-scaled MSE"},
        {"metric": "DSSIM", "category": "Structural & statistical", "characteristics": "Perceptual structure"},
        {"metric": "NMI", "category": "Structural & statistical", "characteristics": "Entropy-based dependence"},
        {"metric": "NCC", "category": "Structural & statistical", "characteristics": "Linear correlation"},
        {"metric": "FHC", "category": "Structural & statistical", "characteristics": "4D hypershell FSC + half-bit"},
    ]


def table_s1_ultrasparse_angles() -> list[dict[str, Any]]:
    """Supplementary Table S1 — four-view angles per experiment (degrees)."""
    rows: list[list[float]] = [
        [0.0, 45.0, 90.0, 135.0],
        [11.25, 56.25, 101.25, 146.25],
        [22.5, 67.5, 112.5, 157.5],
        [33.75, 78.75, 123.75, 168.75],
        [45.0, 90.0, 135.0, 0.0],
        [56.25, 101.25, 146.25, 11.25],
        [67.5, 112.5, 157.5, 22.5],
        [78.75, 123.75, 168.75, 33.75],
        [0.0, 45.0, 90.0, 135.0],
        [11.25, 56.25, 101.25, 146.25],
        [22.5, 67.5, 112.5, 157.5],
        [33.75, 78.75, 123.75, 168.75],
        [45.0, 90.0, 135.0, 0.0],
        [56.25, 101.25, 146.25, 11.25],
        [67.5, 112.5, 157.5, 22.5],
        [78.75, 123.75, 168.75, 33.75],
    ]
    out: list[dict[str, Any]] = []
    for i, angles in enumerate(rows, start=1):
        out.append(
            {
                "experiment": i,
                "angle_1": angles[0],
                "angle_2": angles[1],
                "angle_3": angles[2],
                "angle_4": angles[3],
            }
        )
    return out


def sparse_regime_findings() -> list[dict[str, Any]]:
    """Key qualitative findings from Figure 2 (sparse projection bootstrap)."""
    return [
        {"projections": 2, "note": "Strongly under-determined; correlated biased solutions"},
        {"projections": 4, "note": "Improving agreement with pseudo-reference ỹ"},
        {"projections": 8, "note": "Metrics approach 16-projection baseline; std < 1%"},
    ]


def ultrasparse_regime_findings() -> list[dict[str, Any]]:
    """Key qualitative findings from Figure 3 (experiment-count bootstrap)."""
    return [
        {"experiments": 1, "note": "High variance; under-determined"},
        {"experiments": 4, "note": "Stable performance across metrics"},
        {"experiments": 8, "note": "Close to 16-experiment full-set baseline"},
    ]


def compatible_reconstructors() -> list[str]:
    """Methods cited as compatible with the BCV framework."""
    return [
        "4D-ONIX",
        "NeCT",
        "NeRF-CA",
        "STRT",
        "DYRECT",
        "X2-Gaussian",
    ]
