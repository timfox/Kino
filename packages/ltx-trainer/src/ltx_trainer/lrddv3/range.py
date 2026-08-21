"""Drone range from GPS metadata (Sec. III-D, Eq. 1–5)."""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def haversine_horizontal_m(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
    *,
    earth_radius_m: float = 6_371_000.0,
) -> float:
    """Eq. (1)–(3): horizontal distance between camera and target drones."""
    phi1 = math.radians(lat1_deg)
    phi2 = math.radians(lat2_deg)
    lam1 = math.radians(lon1_deg)
    lam2 = math.radians(lon2_deg)
    a = math.sin((phi1 - phi2) / 2) ** 2
    b = math.cos(phi1) * math.cos(phi2) * math.sin((lam1 - lam2) / 2) ** 2
    return 2.0 * earth_radius_m * math.asin(math.sqrt(a + b))


def vertical_distance_m(alt_camera_m: float, alt_target_m: float) -> float:
    """Eq. (4)."""
    return abs(alt_camera_m - alt_target_m)


def drone_range_m(
    lat_camera: float,
    lon_camera: float,
    alt_camera: float,
    lat_target: float,
    lon_target: float,
    alt_target: float,
    *,
    earth_radius_m: float = 6_371_000.0,
) -> float:
    """Eq. (5): 3D range between camera and target."""
    d_h = haversine_horizontal_m(
        lat_camera, lon_camera, lat_target, lon_target, earth_radius_m=earth_radius_m
    )
    d_v = vertical_distance_m(alt_camera, alt_target)
    return math.sqrt(d_h * d_h + d_v * d_v)


def batch_drone_range_m(
    lat_camera: "Tensor",
    lon_camera: "Tensor",
    alt_camera: "Tensor",
    lat_target: "Tensor",
    lon_target: "Tensor",
    alt_target: "Tensor",
    *,
    earth_radius_m: float = 6_371_000.0,
) -> "Tensor":
    """Vectorized range for smoke tests."""
    import torch

    out = []
    for i in range(lat_camera.shape[0]):
        out.append(
            drone_range_m(
                float(lat_camera[i]),
                float(lon_camera[i]),
                float(alt_camera[i]),
                float(lat_target[i]),
                float(lon_target[i]),
                float(alt_target[i]),
                earth_radius_m=earth_radius_m,
            )
        )
    return torch.tensor(out, dtype=torch.float32)
