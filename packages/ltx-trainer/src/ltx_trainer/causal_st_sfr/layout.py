"""Spatial layouts: Fibonacci sphere quadrature (paper cites González 2010)."""

from __future__ import annotations

import math
from typing import Final

import numpy as np

LIMITATIONS: Final[str] = (
    "Reference stub only: no full O((MW)³) solvers, no DTU download, no image-source simulator wired in-repo."
)


def fibonacci_sphere_points(
    n: int,
    *,
    radius: float,
    center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Nearly uniform points on a sphere of given radius (equal-area style Fibonacci lattice).

    Returns array of shape (n, 3) in Cartesian coordinates.
    """
    if n < 1:
        raise ValueError("n must be positive")
    c = np.asarray(center, dtype=np.float64).reshape(3)
    pts = np.zeros((n, 3), dtype=np.float64)
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio
    for i in range(n):
        z = 1.0 - (2.0 * i + 1.0) / n
        z = float(np.clip(z, -1.0, 1.0))
        theta = 2.0 * math.pi * i / phi
        r_xy = math.sqrt(max(0.0, 1.0 - z * z))
        pts[i, 0] = c[0] + radius * r_xy * math.cos(theta)
        pts[i, 1] = c[1] + radius * r_xy * math.sin(theta)
        pts[i, 2] = c[2] + radius * z
    return pts


def circular_mic_array_xy(
    m: int,
    *,
    radius_m: float,
    center_m: tuple[float, float, float],
    z_m: float,
) -> np.ndarray:
    """M microphones on a circle in the z=z_m plane (paper §VI-A)."""
    cx, cy, _ = center_m
    angles = np.linspace(0.0, 2.0 * np.pi, num=m, endpoint=False)
    x = cx + radius_m * np.cos(angles)
    y = cy + radius_m * np.sin(angles)
    z = np.full(m, z_m, dtype=np.float64)
    return np.stack([x, y, z], axis=1)
