"""Autoregressive Spatial LM + LocDiT streaming stub (Sec. 3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.swansphere.config import SwanSphereConfig
from ltx_trainer.swansphere.foa import FOA_CHANNEL_NAMES


@dataclass
class PatchPlan:
    patch_index: int
    latent_frames: int
    has_video: bool
    has_text: bool


def plan_patches(num_latent_frames: int, cfg: SwanSphereConfig | None = None) -> list[PatchPlan]:
    """Non-overlapping patches for streaming inference."""
    cfg = cfg or SwanSphereConfig()
    plans: list[PatchPlan] = []
    start = 0
    idx = 0
    while start < num_latent_frames:
        end = min(start + cfg.patch_frames, num_latent_frames)
        plans.append(
            PatchPlan(
                patch_index=idx,
                latent_frames=end - start,
                has_video=True,
                has_text=True,
            )
        )
        start += cfg.patch_stride
        idx += 1
    return plans


def spatial_lm_hidden(
    video_token: np.ndarray,
    history: np.ndarray,
    text_token: np.ndarray | None,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Toy semantic condition h_t for current patch."""
    rng = np.random.default_rng(seed)
    parts = [video_token, history]
    if text_token is not None:
        parts.append(text_token)
    stacked = np.concatenate([p.ravel() for p in parts])
    w = rng.standard_normal(stacked.size)
    return (w * 0.01 + stacked * 0.1).astype(np.float64)


def locdit_flow_step(
    h_t: np.ndarray,
    noise: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    """Single flow-matching denoise step toward latent (toy)."""
    target = h_t[: noise.size] if h_t.size >= noise.size else np.pad(h_t, (0, noise.size - h_t.size))
    return noise + step * (target - noise)


def stream_patch_latents(
    num_latent_frames: int,
    *,
    cfg: SwanSphereConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Simulate patch-wise generation with causal history."""
    cfg = cfg or SwanSphereConfig()
    rng = np.random.default_rng(seed)
    dim = cfg.latent_dim
    plans = plan_patches(num_latent_frames, cfg)
    history = np.zeros(dim)
    latents: list[np.ndarray] = []
    for p in plans:
        v_tok = rng.standard_normal(dim)
        t_tok = rng.standard_normal(dim // 2)
        h_t = spatial_lm_hidden(v_tok, history, t_tok, seed=seed + p.patch_index)
        z = rng.standard_normal(dim)
        for _ in range(cfg.locdit_steps):
            z = locdit_flow_step(h_t, z, step=1.0 / cfg.locdit_steps)
        latents.append(z)
        history = 0.5 * history + 0.5 * z
    return {
        "num_patches": len(plans),
        "latent_frames": num_latent_frames,
        "foa_channels": list(FOA_CHANNEL_NAMES),
        "final_latent_norm": float(np.linalg.norm(latents[-1])) if latents else 0.0,
    }


def latency_breakdown(cfg: SwanSphereConfig | None = None) -> dict[str, float]:
    """Paper first-chunk latency decomposition (Sec. 4.2)."""
    cfg = cfg or SwanSphereConfig()
    return {
        "spatial_lm_s": 0.03,
        "locdit_s": 0.14,
        "encode_decode_s": 0.04,
        "first_chunk_total_s": cfg.first_chunk_latency_s,
        "full_stream_s": cfg.total_stream_latency_s,
    }
