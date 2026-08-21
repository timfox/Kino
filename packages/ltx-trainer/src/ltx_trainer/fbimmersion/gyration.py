"""Gyration circles from radar polygons (Sec. 4, Algorithm 1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.fbimmersion.submetrics import radar_angles


@dataclass(frozen=True)
class GyrationCircle:
    """Centroid and radius of gyration for one motion profile."""

    xc: float
    yc: float
    radius: float


def iqr_to_polygon_vertices(iqr_values: Tensor | list[float]) -> Tensor:
    """Convert IQR radar values to Cartesian polygon vertices (Algorithm 1, lines 5–7)."""
    if not isinstance(iqr_values, Tensor):
        iqr_values = torch.tensor(iqr_values, dtype=torch.float64)
    n = iqr_values.numel()
    theta = radar_angles(n)
    x = iqr_values * torch.cos(theta)
    y = iqr_values * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def polygon_area(vertices: Tensor) -> float:
    """Shoelace formula (Algorithm 1, line 13)."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * torch.abs(torch.dot(x, torch.roll(y, -1)) - torch.dot(y, torch.roll(x, -1))).item())


def polygon_centroid(vertices: Tensor) -> tuple[float, float]:
    """xc, yc = mean of vertices (Algorithm 1, lines 9–10)."""
    return float(vertices[:, 0].mean().item()), float(vertices[:, 1].mean().item())


def radius_of_gyration(vertices: Tensor) -> GyrationCircle:
    """r = sqrt(I/A) about polygon centroid (Algorithm 1, lines 11–14)."""
    xc, yc = polygon_centroid(vertices)
    centered = vertices - torch.tensor([xc, yc], dtype=vertices.dtype)
    inertia = float((centered[:, 0] ** 2 + centered[:, 1] ** 2).sum().item())
    area = polygon_area(vertices)
    if area <= 1e-12:
        return GyrationCircle(xc=xc, yc=yc, radius=0.0)
    r = math.sqrt(inertia / area)
    return GyrationCircle(xc=xc, yc=yc, radius=r)


def gyration_circle_from_iqrs(iqr_values: Tensor | list[float]) -> GyrationCircle:
    """Full pipeline: IQR radar chart → gyration circle."""
    verts = iqr_to_polygon_vertices(iqr_values)
    return radius_of_gyration(verts)


def gyration_circle_from_submetrics(submetric_iqr: dict[str, float]) -> GyrationCircle:
    """Build gyration circle preserving submetric dict key order."""
    values = list(submetric_iqr.values())
    return gyration_circle_from_iqrs(values)
