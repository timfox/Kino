"""Compact geometry conditioning sidecars for LTX shards (latent pseudo-depth)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ltx_trainer.fuse_flow.fuse import measurement_confidence
from ltx_trainer.fuse_flow.geometry import pseudo_depth_map


def _target_hw(shape: tuple[int, int], *, max_side: int = 64) -> tuple[int, int]:
    h, w = shape
    scale = min(1.0, max_side / max(h, w))
    nh = max(4, int(h * scale))
    nw = max(4, int(w * scale))
    return nh, nw


def downsample_depth_u8(depth: np.ndarray, *, max_side: int = 64) -> np.ndarray:
    """Normalize depth to uint8 thumbnail for optional shard storage."""
    d = depth.astype(np.float32)
    d = d - d.min()
    d = d / (d.max() + 1e-6)
    t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
    nh, nw = _target_hw(d.shape, max_side=max_side)
    small = F.interpolate(t, size=(nh, nw), mode="area").squeeze().numpy()
    return (small * 255.0).clip(0, 255).astype(np.uint8)


def latent_geometry_conditioning(
    z: np.ndarray,
    *,
    include_depth_u8: bool | None = None,
    max_side: int = 64,
) -> dict[str, Any]:
    """Build JSON-serializable conditioning block from video latents."""
    if include_depth_u8 is None:
        include_depth_u8 = os.environ.get("GOPEX_FUSE_FLOW_DEPTH_U8", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    depth = pseudo_depth_map(z)
    conf = measurement_confidence(torch.from_numpy(depth).float())
    high_conf = float((conf > 0.35).float().mean().item())
    out: dict[str, Any] = {
        "pseudo_depth_hw": [int(depth.shape[0]), int(depth.shape[1])],
        "high_confidence_fraction": round(high_conf, 4),
        "depth_mean": round(float(depth.mean()), 5),
        "depth_std": round(float(depth.std()), 5),
    }
    if include_depth_u8:
        out["pseudo_depth_u8"] = downsample_depth_u8(depth, max_side=max_side)
        nh, nw = out["pseudo_depth_u8"].shape
        out["pseudo_depth_u8_hw"] = [int(nh), int(nw)]
    return out
