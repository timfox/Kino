"""Fold Mirage latent-cache readiness into video latent shards (arXiv:2606.09828)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.mirage.config import MirageConfig
from ltx_trainer.mirage.memory import LatentSpatialMemory
from ltx_trainer.mirage.rollout import default_intrinsics, identity_extrinsics, synthetic_depth


def mirage_meta_block() -> dict[str, Any]:
    return {
        "mirage": {
            "arxiv_id": "2606.09828",
            "fold_role": "latent_spatial_memory_proxy",
        }
    }


def _latent_time_series(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"expected 4D or 5D latents, got shape {arr.shape}")
    if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= 64:
        return arr
    if arr.shape[1] <= 64:
        return np.transpose(arr, (1, 0, 2, 3))
    return arr


def cache_readiness_from_latents(z: np.ndarray, cfg: MirageConfig | None = None) -> tuple[float, int]:
    """Proxy: lift first frame into latent cache and measure readout coverage."""
    cfg = cfg or MirageConfig()
    z = _latent_time_series(z)
    frame = z[0]
    c = frame.shape[0]
    if c != cfg.latent_channels:
        c = min(c, cfg.latent_channels)
        frame = frame[:c]
    h, w = frame.shape[1], frame.shape[2]
    mem = LatentSpatialMemory(channels=c)
    depth = synthetic_depth(cfg.rgb_height, cfg.rgb_width, seed=7)
    k = default_intrinsics(cfg)
    mem.add_from_latent(frame, depth, k, identity_extrinsics(), rgb_hw=cfg.rgb_grid)
    _, vis = mem.readout(k, identity_extrinsics(yaw=0.1), rgb_hw=cfg.rgb_grid, latent_hw=(h, w))
    return float(vis.mean()), len(mem)


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(mirage_meta_block())
    latents = data.get("latents")
    if latents is None:
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    coverage, n_pts = cache_readiness_from_latents(arr)
    meta = data.get("meta") or {}
    if meta.get("has_depth_prior"):
        coverage = float(min(1.0, coverage * 1.08 + 0.02))
    out["mirage"].update(
        {
            "readout_coverage": round(coverage, 4),
            "cache_points": n_pts,
            "has_depth_prior": bool(meta.get("has_depth_prior")),
        }
    )
    return out
