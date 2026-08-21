"""Reference frames, FOV, and geometry-based visibility (§2.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ObjectState:
    """Global object pose (center + extent)."""

    object_id: int
    x: float
    y: float
    z: float
    length: float = 4.0
    width: float = 2.0
    height: float = 1.5
    class_name: str = "car"


@dataclass(frozen=True)
class SensorCalibration:
    """Rigid sensor-from-world transform (yaw-pitch-roll + translation)."""

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    fov_deg: float = 90.0
    image_w: int = 1920
    image_h: int = 1080


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def world_to_sensor(obj: ObjectState, cal: SensorCalibration) -> np.ndarray:
    """Transform global center to sensor frame (AVstack reference-chain stub)."""
    p = np.array([obj.x - cal.x, obj.y - cal.y, obj.z - cal.z], dtype=np.float64)
    r = _rotation_matrix(cal.yaw, cal.pitch, cal.roll)
    return r.T @ p


def in_field_of_view(p_sensor: np.ndarray, cal: SensorCalibration) -> bool:
    if p_sensor[2] <= 0.1:
        return False
    half = math.radians(cal.fov_deg * 0.5)
    az = math.atan2(p_sensor[0], p_sensor[2])
    el = math.atan2(p_sensor[1], p_sensor[2])
    return abs(az) <= half and abs(el) <= half


def project_to_image(p_sensor: np.ndarray, cal: SensorCalibration, focal: float = 800.0) -> tuple[float, float] | None:
    if p_sensor[2] <= 0.1:
        return None
    u = focal * p_sensor[0] / p_sensor[2] + cal.image_w * 0.5
    v = focal * p_sensor[1] / p_sensor[2] + cal.image_h * 0.5
    if 0 <= u < cal.image_w and 0 <= v < cal.image_h:
        return float(u), float(v)
    return None


def visibility_from_depth(
    expected_depth: float,
    depth_map: np.ndarray,
    u: int,
    v: int,
    *,
    tolerance: float = 1.5,
) -> str:
    """Geometry-based occlusion: compare ray depth to measured depth (§2.2)."""
    if u < 0 or v < 0 or u >= depth_map.shape[1] or v >= depth_map.shape[0]:
        return "unknown"
    measured = float(depth_map[v, u])
    if measured <= 0 or not np.isfinite(measured):
        return "unknown"
    if measured + tolerance < expected_depth:
        return "occluded"
    return "visible"


def label_objects_for_sensor(
    objects: Iterable[ObjectState],
    cal: SensorCalibration,
    depth_map: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    """Per-sensor labels after FOV + optional occlusion filter."""
    labels: list[dict[str, float | int | str]] = []
    for obj in objects:
        p = world_to_sensor(obj, cal)
        if not in_field_of_view(p, cal):
            continue
        vis = "visible"
        if depth_map is not None:
            uv = project_to_image(p, cal)
            if uv is None:
                continue
            u, v = int(uv[0]), int(uv[1])
            vis = visibility_from_depth(float(p[2]), depth_map, u, v)
            if vis == "occluded":
                continue
        uv = project_to_image(p, cal)
        if uv is None:
            continue
        labels.append(
            {
                "object_id": obj.object_id,
                "class": obj.class_name,
                "u": uv[0],
                "v": uv[1],
                "depth_m": float(p[2]),
                "visibility": vis,
            }
        )
    return labels
