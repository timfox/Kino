"""Mirage rollout pseudocode (Algorithm 1, arXiv:2606.09828)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator

import numpy as np

from ltx_trainer.mirage.config import MirageConfig
from ltx_trainer.mirage.geometry import admissible_lambda
from ltx_trainer.mirage.memory import LatentSpatialMemory
from ltx_trainer.mirage.segment import build_update_mask


class RolloutPhase(str, Enum):
    """High-level steps in Algorithm 1."""

    INIT_ENCODE = "init_encode"
    INIT_DEPTH = "init_depth"
    INIT_SEGMENT = "init_segment"
    INIT_CACHE = "init_cache"
    READ_CHUNK = "read_chunk"
    DENOISE = "denoise"
    DECODE_UPDATE = "decode_update"


@dataclass
class Algorithm1State:
    """Mutable state while executing Algorithm 1."""

    memory: LatentSpatialMemory
    tau: int = 0
    output_frames: list[dict[str, Any]] = field(default_factory=list)
    phases: list[RolloutPhase] = field(default_factory=list)


def algorithm1_steps(cfg: MirageConfig | None = None) -> list[str]:
    """Human-readable step list matching the paper (lines 1–21)."""
    cfg = cfg or MirageConfig()
    chunk = cfg.chunk_latent_frames
    return [
        "Require: initial frame I0; camera trajectory {(Et, Kt)} with E0 fixed",
        "z0 <- E(I0)",
        "D0 <- DepthAnything3(I0)",
        "M0 <- SAM3(Qwen3-VL(I0)) ∪ sky(I0)",
        f"M <- lift (z0, D0, K0, E0) via Eq. 4 on admissible Λ0",
        "τ <- 0; O <- {I0}",
        "while τ < T:",
        f"  sample latent chunk W = {{τ+1, …, τ+|W|}} (|W|={chunk})",
        "  for t in W:  z_hat_t, m_t <- readout(M, Et, Kt) via Eq. 5",
        "  {zt}_t <- backbone({z_hat_t, m_t}, preceding, reference)",
        "  for t in W: decode It; depth; segment; z_tilde <- E(It); M <- M ∪ Λ_t via Eq. 6",
        "  τ <- τ + |W|",
        "return O",
    ]


def iter_rollout_phases(
    cfg: MirageConfig,
    *,
    n_chunks: int,
    seed: int = 0,
) -> Iterator[tuple[RolloutPhase, dict[str, Any]]]:
    """Yield phase tags for a synthetic Algorithm 1 trace (CPU stub)."""
    c, h, w = cfg.latent_channels, cfg.latent_height, cfg.latent_width

    yield RolloutPhase.INIT_ENCODE, {"z0_shape": (c, h, w)}
    yield RolloutPhase.INIT_DEPTH, {"depth_hw": cfg.rgb_grid}
    yield RolloutPhase.INIT_SEGMENT, {"mask_cells": int(build_update_mask(h, w, seed=seed).sum())}
    yield RolloutPhase.INIT_CACHE, {"lambda0_size": h * w}

    for chunk_idx in range(n_chunks):
        yield RolloutPhase.READ_CHUNK, {"chunk": chunk_idx, "views": cfg.chunk_latent_frames - 1}
        yield RolloutPhase.DENOISE, {"chunk": chunk_idx, "flow_steps": cfg.flow_steps}
        yield RolloutPhase.DECODE_UPDATE, {"chunk": chunk_idx, "overlap": cfg.chunk_overlap_latent_frames}


def run_algorithm1_stub(
    cfg: MirageConfig | None = None,
    *,
    n_chunks: int = 2,
    seed: int = 0,
    depth_fn: Callable[[int, int, int], np.ndarray] | None = None,
    latent_fn: Callable[[int, int, int, int], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Minimal executable trace of Algorithm 1 using synthetic tensors."""
    cfg = cfg or MirageConfig()
    c, h, w = cfg.latent_channels, cfg.latent_height, cfg.latent_width
    rgb_hw = cfg.rgb_grid

    def _depth(s: int) -> np.ndarray:
        if depth_fn is not None:
            return depth_fn(*rgb_hw, s)
        yy, xx = np.mgrid[0 : rgb_hw[0], 0 : rgb_hw[1]]
        return np.clip(2.0 + 0.003 * xx + 0.002 * yy + 0.01 * s, 0.5, 8.0)

    def _latent(s: int) -> np.ndarray:
        if latent_fn is not None:
            return latent_fn(c, h, w, s)
        rng = np.random.default_rng(s)
        return rng.standard_normal((c, h, w))

    fx = fy = 0.9 * cfg.rgb_width
    k = np.array([[fx, 0, cfg.rgb_width / 2], [0, fy, cfg.rgb_height / 2], [0, 0, 1]], dtype=np.float64)
    e0 = np.eye(4, dtype=np.float64)
    mem = LatentSpatialMemory(channels=c)
    state = Algorithm1State(memory=mem)

    z0 = _latent(seed)
    d0 = _depth(seed)
    seg0 = build_update_mask(h, w, seed=seed)
    lam0 = admissible_lambda(d0, rgb_hw, dynamic_mask=seg0)
    state.phases.append(RolloutPhase.INIT_ENCODE)
    mem.add_from_latent(z0, d0, k, e0, rgb_hw=rgb_hw, dynamic_mask=seg0)
    state.output_frames.append({"tau": 0, "role": "I0"})
    state.tau = 0

    chunk_size = cfg.chunk_latent_frames - cfg.chunk_overlap_latent_frames
    for chunk_idx in range(n_chunks):
        state.phases.append(RolloutPhase.READ_CHUNK)
        for t in range(1, chunk_size + 1):
            yaw = 0.12 * (chunk_idx + 1) * t / chunk_size
            et = np.array(
                [[np.cos(yaw), 0, np.sin(yaw), 0.05 * t], [0, 1, 0, 0], [-np.sin(yaw), 0, np.cos(yaw), 0], [0, 0, 0, 1]],
                dtype=np.float64,
            )
            z_hat, vis = mem.readout(k, et, rgb_hw=rgb_hw, latent_hw=(h, w))
            state.phases.append(RolloutPhase.DENOISE)
            zt = z_hat + 0.05 * _latent(seed + 1000 + chunk_idx * 10 + t)
            dt = _depth(seed + 2000 + chunk_idx * 10 + t)
            seg = build_update_mask(h, w, seed=seed + chunk_idx + t)
            _ = admissible_lambda(dt, rgb_hw, dynamic_mask=seg)
            mem.update_union(zt, dt, k, et, rgb_hw=rgb_hw, dynamic_mask=seg)
            state.output_frames.append({"tau": state.tau + t, "chunk": chunk_idx, "visibility": float(vis.mean())})
            state.phases.append(RolloutPhase.DECODE_UPDATE)
        state.tau += chunk_size

    return {
        "algorithm": "Mirage-Algorithm-1",
        "n_chunks": n_chunks,
        "memory_size": len(mem),
        "output_frames": len(state.output_frames),
        "lambda0_cells": len(lam0),
        "phases_executed": [p.value for p in state.phases],
        "steps": algorithm1_steps(cfg),
    }
