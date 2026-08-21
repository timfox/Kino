"""GPS telemetry → image-space trajectories (equirectangular local EN, Sec. III-C1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.traj_i2v.config import MLAT_M_PER_DEG, TrajI2VConfig


@dataclass
class GpsFix:
    """Single ASV log row."""

    t_s: float
    lon: float
    lat: float
    vessel_id: int = 0


@dataclass
class VesselAnchor:
    """Manual reference click + metric scale for one vessel."""

    vessel_id: int
    cx: float
    cy: float
    lon0: float
    lat0: float
    t0_s: float


def lonlat_to_en(
    lon: float,
    lat: float,
    origin_lon: float,
    origin_lat: float,
) -> tuple[float, float]:
    """Eq. (1–2): equirectangular East–North metres."""
    mlat = MLAT_M_PER_DEG
    mlon = mlat * math.cos(math.radians(origin_lat))
    e = (lon - origin_lon) * mlon
    n = (lat - origin_lat) * mlat
    return e, n


def rotate_en_to_image_axes(e: float, n: float, yaw_deg: float) -> tuple[float, float]:
    """Eq. (3): align metric frame with image axes using estimated camera yaw."""
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    fx = c * e - s * n
    fy = s * e + c * n
    return fx, fy


def estimate_scale_px_per_m(
    anchor_a: VesselAnchor,
    anchor_b: VesselAnchor,
    origin_lon: float,
    origin_lat: float,
) -> float:
    """Eq. (4): inter-vessel pixel distance / metric distance at t0."""
    ea, na = lonlat_to_en(anchor_a.lon0, anchor_a.lat0, origin_lon, origin_lat)
    eb, nb = lonlat_to_en(anchor_b.lon0, anchor_b.lat0, origin_lon, origin_lat)
    dm = math.hypot(eb - ea, nb - na)
    if dm < 1e-6:
        raise ValueError("Vessel anchors too close in GPS for scale estimation")
    dp = math.hypot(anchor_b.cx - anchor_a.cx, anchor_b.cy - anchor_a.cy)
    return dp / dm


def project_gps_to_pixel(
    lon: float,
    lat: float,
    anchor: VesselAnchor,
    *,
    origin_lon: float,
    origin_lat: float,
    scale_px_per_m: float,
    yaw_deg: float,
) -> tuple[float, float]:
    """Eq. (5): GPS fix → pixel (x, y) relative to manual click at t0."""
    e, n = lonlat_to_en(lon, lat, origin_lon, origin_lat)
    e0, n0 = lonlat_to_en(anchor.lon0, anchor.lat0, origin_lon, origin_lat)
    fx, fy = rotate_en_to_image_axes(e - e0, n - n0, yaw_deg)
    x = anchor.cx + fx * scale_px_per_m
    y = anchor.cy - fy * scale_px_per_m
    return x, y


def interpolate_log_position(
    fixes: list[GpsFix],
    t_s: float,
) -> tuple[float, float]:
    """Linear interpolation of (lon, lat) along sorted GPS fixes."""
    if not fixes:
        raise ValueError("empty GPS fix list")
    fixes = sorted(fixes, key=lambda f: f.t_s)
    if t_s <= fixes[0].t_s:
        return fixes[0].lon, fixes[0].lat
    if t_s >= fixes[-1].t_s:
        return fixes[-1].lon, fixes[-1].lat
    for i in range(len(fixes) - 1):
        a, b = fixes[i], fixes[i + 1]
        if a.t_s <= t_s <= b.t_s:
            u = (t_s - a.t_s) / max(b.t_s - a.t_s, 1e-9)
            lon = a.lon + u * (b.lon - a.lon)
            lat = a.lat + u * (b.lat - a.lat)
            return lon, lat
    return fixes[-1].lon, fixes[-1].lat


def vessel_trajectory_pixels(
    fixes: list[GpsFix],
    anchor: VesselAnchor,
    *,
    cfg: TrajI2VConfig,
    origin_lon: float,
    origin_lat: float,
    scale_px_per_m: float,
    t_start_s: float,
) -> Tensor:
    """Per-frame pixel centres ``[N, 2]`` for one vessel."""
    pts: list[list[float]] = []
    for i in range(cfg.num_frames):
        t = t_start_s + (i + 1) / cfg.fps
        lon, lat = interpolate_log_position(fixes, t + cfg.log_time_offset_s)
        x, y = project_gps_to_pixel(
            lon,
            lat,
            anchor,
            origin_lon=origin_lon,
            origin_lat=origin_lat,
            scale_px_per_m=scale_px_per_m,
            yaw_deg=cfg.camera_yaw_deg,
        )
        pts.append([x, y])
    return torch.tensor(pts, dtype=torch.float32)


def mean_origin(fixes: list[GpsFix]) -> tuple[float, float]:
    lons = [f.lon for f in fixes]
    lats = [f.lat for f in fixes]
    return sum(lons) / len(lons), sum(lats) / len(lats)
