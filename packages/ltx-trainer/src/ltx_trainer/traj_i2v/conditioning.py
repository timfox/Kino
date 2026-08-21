"""SG-I2V conditioning payload: boxes + trajectories + corner anchors (Sec. III-C1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.traj_i2v.config import DEFAULT_VESSEL_COLORS, TrajI2VConfig
from ltx_trainer.traj_i2v.gps_mapping import GpsFix, VesselAnchor, mean_origin, vessel_trajectory_pixels


@dataclass
class TrajectoryObject:
    """One SG-I2V object descriptor (b_i, T_i)."""

    name: str
    box: Tensor  # [4] x,y,w,h
    trajectory: Tensor  # [N, 2]


@dataclass
class SgI2VConditioning:
    """Six-object conditioning set from the paper."""

    objects: list[TrajectoryObject]
    reference_frame_annotated: Tensor | None = None  # [3,H,W] optional RGB overlay


def box_from_center(cx: float, cy: float, size: float) -> Tensor:
    half = size / 2.0
    return torch.tensor([cx - half, cy - half, size, size], dtype=torch.float32)


def corner_anchors(cfg: TrajI2VConfig) -> list[TrajectoryObject]:
    """Four fixed corner boxes signalling zero global camera motion."""
    w, h = cfg.image_width, cfg.image_height
    inset = cfg.corner_inset_px
    sz = float(cfg.corner_box_size)
    centers = {
        "top_left": (inset + sz / 2, inset + sz / 2),
        "top_right": (w - inset - sz / 2, inset + sz / 2),
        "bottom_left": (inset + sz / 2, h - inset - sz / 2),
        "bottom_right": (w - inset - sz / 2, h - inset - sz / 2),
    }
    static = torch.zeros(cfg.num_frames, 2)
    out: list[TrajectoryObject] = []
    for name, (cx, cy) in centers.items():
        traj = static.clone()
        traj[:, 0] = cx
        traj[:, 1] = cy
        out.append(TrajectoryObject(name=name, box=box_from_center(cx, cy, sz), trajectory=traj))
    return out


def build_vessel_objects(
    log_by_vessel: dict[int, list[GpsFix]],
    anchors: list[VesselAnchor],
    *,
    cfg: TrajI2VConfig,
    scale_px_per_m: float,
    t_start_s: float,
    box_size: float = 48.0,
) -> list[TrajectoryObject]:
    all_fixes = [f for fixes in log_by_vessel.values() for f in fixes]
    origin_lon, origin_lat = mean_origin(all_fixes)
    objects: list[TrajectoryObject] = []
    for anchor in anchors:
        fixes = log_by_vessel.get(anchor.vessel_id, [])
        traj = vessel_trajectory_pixels(
            fixes,
            anchor,
            cfg=cfg,
            origin_lon=origin_lon,
            origin_lat=origin_lat,
            scale_px_per_m=scale_px_per_m,
            t_start_s=t_start_s,
        )
        name = DEFAULT_VESSEL_COLORS.get(anchor.vessel_id, f"vessel_{anchor.vessel_id}")
        objects.append(
            TrajectoryObject(
                name=name,
                box=box_from_center(anchor.cx, anchor.cy, box_size),
                trajectory=traj,
            )
        )
    return objects


def draw_trajectory_overlay(
    frame: Tensor,
    objects: list[TrajectoryObject],
    *,
    arrow_scale: float = 1.0,
) -> Tensor:
    """Simple arrow overlay on ``frame`` [3,H,W] in [0,1] (for visualization / stub I2V input)."""
    out = frame.clone()
    _, h, w = out.shape
    for obj in objects:
        if obj.name.startswith("vessel_") or obj.name in ("green", "yellow"):
            pts = obj.trajectory
            for i in range(pts.shape[0] - 1):
                p0 = pts[i]
                p1 = pts[i + 1]
                dx = (p1[0] - p0[0]) * arrow_scale
                dy = (p1[1] - p0[1]) * arrow_scale
                cx = int(p0[0].item())
                cy = int(p0[1].item())
                if 0 <= cx < w and 0 <= cy < h:
                    color = torch.tensor([0.2, 0.9, 0.3] if obj.name == "green" else [0.95, 0.85, 0.1])
                    out[:, cy, cx] = color
                cx2 = int((p0[0] + dx * 0.5).item())
                cy2 = int((p0[1] + dy * 0.5).item())
                if 0 <= cx2 < w and 0 <= cy2 < h:
                    out[:, cy2, cx2] = color * 0.8
    return out.clamp(0, 1)


def build_sg_i2v_conditioning(
    reference_frame: Tensor,
    log_by_vessel: dict[int, list[GpsFix]],
    anchors: list[VesselAnchor],
    *,
    cfg: TrajI2VConfig,
    scale_px_per_m: float,
    t_start_s: float = 0.0,
    box_size: float = 48.0,
) -> SgI2VConditioning:
    """Assemble six SG-I2V entries: two vessels + four corners."""
    vessels = build_vessel_objects(
        log_by_vessel,
        anchors,
        cfg=cfg,
        scale_px_per_m=scale_px_per_m,
        t_start_s=t_start_s,
        box_size=box_size,
    )
    corners = corner_anchors(cfg)
    objects = vessels + corners
    annotated = None
    if cfg.draw_trajectory_arrows:
        annotated = draw_trajectory_overlay(reference_frame, objects)
    return SgI2VConditioning(objects=objects, reference_frame_annotated=annotated)
