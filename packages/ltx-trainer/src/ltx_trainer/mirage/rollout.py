"""Mirage autoregressive rollout (Algorithm 1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ltx_trainer.mirage.config import MirageConfig
from ltx_trainer.mirage.controlnet import pack_control_input, side_branch_forward
from ltx_trainer.mirage.memory import LatentSpatialMemory
from ltx_trainer.mirage.segment import build_update_mask


def default_intrinsics(cfg: MirageConfig) -> np.ndarray:
    fx = fy = 0.9 * cfg.rgb_width
    cx, cy = cfg.rgb_width / 2.0, cfg.rgb_height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def identity_extrinsics(yaw: float = 0.0, tx: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [c, 0, s, tx],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def synthetic_depth(h: int, w: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 2.0 + 0.003 * xx + 0.002 * yy
    noise = rng.normal(0, 0.02, size=(h, w))
    return np.clip(base + noise, 0.5, 8.0)


def synthetic_latent(c: int, h: int, w: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((c, h, w))


@dataclass
class RolloutState:
    memory: LatentSpatialMemory
    output_frames: list[dict[str, Any]] = field(default_factory=list)
    readout_hits: list[float] = field(default_factory=list)


def run_toy_rollout(
    cfg: MirageConfig | None = None,
    *,
    n_chunks: int = 3,
    chunk_frames: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """CPU stub of Algorithm 1 with synthetic latents and depth."""
    cfg = cfg or MirageConfig()
    c, h, w = cfg.latent_channels, cfg.latent_height, cfg.latent_width
    rgb_hw = cfg.rgb_grid
    k = default_intrinsics(cfg)
    mem = LatentSpatialMemory(channels=c)
    state = RolloutState(memory=mem)

    z0 = synthetic_latent(c, h, w, seed=seed)
    d0 = synthetic_depth(*rgb_hw, seed=seed)
    e0 = identity_extrinsics(0.0, 0.0)
    init_mask = build_update_mask(h, w, seed=seed, exclude_sky=True, exclude_dynamic=True)
    mem.add_from_latent(z0, d0, k, e0, rgb_hw=rgb_hw, dynamic_mask=init_mask)
    state.output_frames.append({"frame": 0, "view": "init"})

    chunk = chunk_frames or min(4, cfg.chunk_latent_frames - 1)
    for chunk_idx in range(n_chunks):
        yaw = 0.15 * (chunk_idx + 1)
        for t in range(1, chunk + 1):
            et = identity_extrinsics(yaw * t / chunk, tx=0.05 * t)
            z_hat, vis = mem.readout(k, et, rgb_hw=rgb_hw, latent_hw=(h, w))
            state.readout_hits.append(float(vis.mean()))
            _ = pack_control_input(z_hat, vis)
            _ = side_branch_forward(pack_control_input(z_hat, vis), seed=seed + chunk_idx + t)
            zt = z_hat + 0.05 * synthetic_latent(c, h, w, seed=seed + 100 + chunk_idx + t)
            dt = synthetic_depth(*rgb_hw, seed=seed + 200 + chunk_idx + t)
            update_mask = build_update_mask(h, w, seed=seed + chunk_idx + t)
            mem.update_union(zt, dt, k, et, rgb_hw=rgb_hw, dynamic_mask=update_mask)
            state.output_frames.append({"frame": len(state.output_frames), "chunk": chunk_idx, "view": t})
    return {
        "n_chunks": n_chunks,
        "memory_size": len(mem),
        "mean_visibility": float(np.mean(state.readout_hits)) if state.readout_hits else 0.0,
        "cache_bytes": mem.footprint_bytes(),
        "output_frames": len(state.output_frames),
    }
