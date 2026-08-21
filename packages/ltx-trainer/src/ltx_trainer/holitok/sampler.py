"""AR+DiT latent generation sampler (HoliTok downstream)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.holitok.ar_dit import generation_objective, patchify_latents, understanding_ce_loss
from ltx_trainer.holitok.config import HoliTokConfig
from ltx_trainer.holitok.encoder import decode, encode


def euler_sample(
    z0: np.ndarray,
    *,
    steps: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Toy Euler flow on latent patches."""
    rng = np.random.default_rng(seed)
    z = np.asarray(z0, dtype=np.float64).copy()
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps
        v = rng.standard_normal(z.shape) * (1.0 - t) + (z0 - z) * t
        z = z + dt * v
    return z


def sample_utterance(
    prompt_tokens: np.ndarray | None = None,
    *,
    duration_s: float = 2.0,
    cfg: HoliTokConfig | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    cfg = cfg or HoliTokConfig()
    rng = np.random.default_rng(seed)
    n_frames = max(1, int(cfg.latent_frame_rate_hz * duration_s))
    z0 = rng.standard_normal((n_frames, cfg.latent_dim)) * 0.1
    if prompt_tokens is not None:
        pt = np.asarray(prompt_tokens, dtype=np.float64)
        z0[: min(n_frames, pt.shape[0])] += pt[:n_frames] * 0.05
    z1 = euler_sample(z0, steps=getattr(cfg, "flow_steps", 8), seed=seed)
    patches = patchify_latents(z1, patch_size=cfg.patch_size)
    wave = decode(z1, cfg=cfg, seed=seed)
    vel_tgt = z1 - z0
    gen = generation_objective(z1 - z0, vel_tgt, np.array([0.2]), np.array([0.0]))
    ce = understanding_ce_loss(rng.standard_normal((patches.shape[0], 32)), np.array([1, 2, 3]))
    return {"latent_frames": n_frames, "wave_samples": int(wave.size), "generation_total": gen["total"], "understanding_ce": ce}


def sampler_smoke(cfg: HoliTokConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or HoliTokConfig()
    sr = cfg.sample_rate_hz
    enc = encode(np.sin(2 * np.pi * 440 * np.arange(int(sr)) / sr), cfg=cfg)
    out = sample_utterance(enc["latent"][:4], duration_s=1.0, cfg=cfg, seed=seed)
    return {"generation_finite": np.isfinite(out["generation_total"]), **out}
