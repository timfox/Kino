"""Scene proxy maps G — depth, optical flow, perspective (Sec. 3.2, Fig. 5).

These stand in for RAFT / DepthAnything / GeoCalib outputs. Downstream VAE stacks
concatenate or encode them; here we only validate shapes and ablation masking (Table 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

StreamName = Literal["depth", "flow", "perspective", "rgb"]


@dataclass
class SceneProxyMaps:
    """Per-frame optional tensors ``[T, …]`` (batch over time for video)."""

    depth: Tensor | None = None  # [T, H, W]
    optical_flow: Tensor | None = None  # [T, 2, H, W]
    perspective: Tensor | None = None  # [T, C_p, H, W] — e.g. normals / field
    source_rgb: Tensor | None = None  # [T, 3, H, W]

    def streams(self) -> dict[str, Tensor | None]:
        return {
            "depth": self.depth,
            "flow": self.optical_flow,
            "perspective": self.perspective,
            "rgb": self.source_rgb,
        }


def mask_proxy_streams(maps: SceneProxyMaps, disabled: set[StreamName]) -> SceneProxyMaps:
    """Zero ablated streams (Table 4 proxy stream contribution)."""
    d = None if "depth" in disabled or maps.depth is None else maps.depth
    f = None if "flow" in disabled or maps.optical_flow is None else maps.optical_flow
    p = None if "perspective" in disabled or maps.perspective is None else maps.perspective
    r = None if "rgb" in disabled or maps.source_rgb is None else maps.source_rgb
    return SceneProxyMaps(depth=d, optical_flow=f, perspective=p, source_rgb=r)


def assert_proxy_shapes(
    maps: SceneProxyMaps,
    *,
    t: int,
    h: int,
    w: int,
) -> None:
    """Raise if provided streams disagree on ``(T,H,W)``."""
    for name, ten in maps.streams().items():
        if ten is None:
            continue
        if ten.dim() < 3:
            raise ValueError(f"{name}: expected at least 3 dims, got {tuple(ten.shape)}")
        if ten.shape[0] != t:
            raise ValueError(f"{name}: T mismatch {ten.shape[0]} vs {t}")
        if ten.shape[-2] != h or ten.shape[-1] != w:
            raise ValueError(f"{name}: spatial mismatch {tuple(ten.shape)} vs H={h} W={w}")
