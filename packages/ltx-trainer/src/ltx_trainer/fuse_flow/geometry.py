"""Latent-space geometry proxies for FUSE-Flow fold + LTX training (no GPU depth required)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

_MULTI_VIEW_PATH = re.compile(
    r"(cam\d|camera[_-]?\d|view[_-]?\d|multi[_-]?view|rig[_-]?\d|_v\d{1,2}[_./])",
    re.I,
)
_MULTI_VIEW_CAPTION = re.compile(
    r"\b(multi[- ]?camera|multi[- ]?view|synchronized cameras?|camera array|rig)\b",
    re.I,
)


def latent_frame_series(arr: np.ndarray) -> np.ndarray:
    """Return ``[T, C, H, W]`` from 4D/5D latent tensors."""
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"expected 4D or 5D latents, got shape {arr.shape}")
    if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= 64:
        return arr
    if arr.shape[1] <= 64:
        return np.transpose(arr, (1, 0, 2, 3))
    return arr


def pseudo_depth_map(z: np.ndarray) -> np.ndarray:
    """Single-channel pseudo-depth from latent magnitude (middle frame)."""
    z = latent_frame_series(z)
    mid = z[min(z.shape[0] // 2, z.shape[0] - 1)]
    depth = np.abs(mid).mean(axis=0)
    depth = depth - depth.min()
    return depth + 1e-3


def spatial_coherence_score(z: np.ndarray) -> float:
    """High when spatial structure is smooth (fusion-friendly latent layout)."""
    depth = pseudo_depth_map(z)
    gx = np.abs(np.diff(depth, axis=1)).mean()
    gy = np.abs(np.diff(depth, axis=0)).mean()
    grad = float(gx + gy)
    return float(np.clip(1.0 / (1.0 + grad * 8.0), 0.0, 1.0))


def temporal_stability_score(z: np.ndarray) -> float:
    """Inverse temporal jerk on per-frame energy (complements phyworld)."""
    z = latent_frame_series(z)
    if z.shape[0] < 3:
        return 0.55
    energy = z.reshape(z.shape[0], -1).std(axis=1)
    vel = np.diff(energy)
    jerk = np.diff(vel)
    jerk_std = float(jerk.std()) if jerk.size else 0.0
    return float(np.clip(1.0 / (1.0 + jerk_std * 3.0), 0.0, 1.0))


def geometry_stability_proxy(z: np.ndarray) -> tuple[float, float]:
    """``(geometry_stability_0_1, temporal_jerk_std)`` for fold sidecars."""
    z = latent_frame_series(z)
    spatial = spatial_coherence_score(z)
    temporal = temporal_stability_score(z)
    jerk_std = 0.0
    if z.shape[0] >= 3:
        energy = z.reshape(z.shape[0], -1).std(axis=1)
        vel = np.diff(energy)
        jerk = np.diff(vel)
        jerk_std = float(jerk.std()) if jerk.size else 0.0
    stability = float(np.clip(0.55 * spatial + 0.45 * temporal, 0.0, 1.0))
    return stability, jerk_std


def fusion_readiness_proxy(z: np.ndarray, *, high_conf_threshold: float = 0.35) -> float:
    """Fraction of pseudo-depth pixels above MCM-style confidence (FUSE readiness)."""
    import torch
    import torch.nn.functional as F

    from ltx_trainer.fuse_flow.fuse import measurement_confidence

    depth = torch.from_numpy(pseudo_depth_map(z)).float()
    conf = measurement_confidence(depth)
    frac = float((conf > high_conf_threshold).float().mean().item())
    spatial = spatial_coherence_score(z)
    return float(np.clip(0.5 * frac + 0.5 * spatial, 0.0, 1.0))


def multi_view_hint_from_meta(meta: dict[str, Any], *, path: str = "", caption: str = "") -> tuple[bool, int]:
    """Detect rig / multi-camera clips from shard metadata or naming."""
    n_cam = int(meta.get("n_cameras") or meta.get("num_cameras") or 0)
    if n_cam >= 2:
        return True, n_cam
    if meta.get("multi_view") or meta.get("multi_camera") or meta.get("rig_id"):
        return True, max(n_cam, 2)
    path_hit = bool(path and _MULTI_VIEW_PATH.search(path))
    cap_hit = bool(caption and _MULTI_VIEW_CAPTION.search(caption))
    if path_hit or cap_hit:
        return True, 2
    return False, max(n_cam, 1)
