"""Synthetic indoor scene + 3DGS-style render stub (Sec. 3.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class ObjectInstance:
    label: str
    description: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass
class SceneAnnotation:
    scene_id: str
    objects: list[ObjectInstance] = field(default_factory=list)
    camera_positions: list[tuple[float, float, float]] = field(default_factory=list)


def synthetic_scannet_scene(*, scene_id: str = "demo_001") -> SceneAnnotation:
    """Minimal ScanNet++-style annotation for tests."""
    return SceneAnnotation(
        scene_id=scene_id,
        objects=[
            ObjectInstance("chair", "black modern chair with angular base", (1.2, 0.5, 2.0), (0.5, 0.9, 0.5)),
            ObjectInstance("door", "wooden door with glass window", (2.5, 1.0, 3.0), (0.1, 2.0, 0.9)),
            ObjectInstance("monitor", "flat screen monitor on desk", (0.8, 0.9, 1.5), (0.5, 0.4, 0.05)),
        ],
        camera_positions=[(0.0, 1.5, 0.0), (0.5, 1.5, 0.8)],
    )


def render_clean_view(
    scene: SceneAnnotation,
    *,
    view_idx: int = 0,
    height: int = 128,
    width: int = 128,
) -> tuple[Tensor, Tensor]:
    """Return ``(rgb [3,H,W], depth [H,W])`` proxy render."""
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height),
        torch.linspace(0, 1, width),
        indexing="ij",
    )
    depth = (1.0 - yy * 0.8).clamp(0.05, 1.0)
    wall = torch.stack([0.85 - yy * 0.2, 0.82 - yy * 0.15, 0.78 - yy * 0.1], dim=0)
    floor = torch.stack([0.35 + yy * 0.1, 0.32 + yy * 0.08, 0.30 + yy * 0.06], dim=0)
    rgb = wall * (xx > 0.3).float() + floor * (xx <= 0.3).float()
    cam = scene.camera_positions[view_idx % len(scene.camera_positions)]
    for obj in scene.objects:
        ox = 0.3 + 0.4 * (obj.center[0] / 4.0) + cam[2] * 0.05
        oy = 0.2 + 0.5 * (obj.center[1] / 2.5)
        blob = torch.exp(-(((xx - ox) ** 2 + (yy - oy) ** 2) / 0.008))
        tint = torch.tensor([0.2, 0.25, 0.35]).view(3, 1, 1)
        rgb = rgb * (1.0 - blob) + tint * blob
    return rgb.clamp(0.0, 1.0), depth
