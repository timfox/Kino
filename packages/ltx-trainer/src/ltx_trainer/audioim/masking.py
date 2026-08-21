"""Latent masking for flow-matching prompt conditioning (§2.1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.audioim.config import AudioImConfig


def split_latent_mask(
    latent: np.ndarray,
    *,
    prompt_ratio: float = 0.375,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split VAE latent into prompt x_p and target x_1 regions."""
    x = np.asarray(latent, dtype=np.float64)
    n = x.shape[-1]
    split = int(n * prompt_ratio)
    m = np.zeros(n, dtype=np.float64)
    m[:split] = 1.0
    xp = x * m
    x1 = x * (1.0 - m)
    return xp, x1, m


def flow_interpolate(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """x_t = t * x_1 + (1 - t) * x_0."""
    return t * x1 + (1.0 - t) * x0


def flow_matching_loss(
    velocity: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
) -> float:
    """||v_theta - (x1 - x0)||^2 stub."""
    target = x1 - x0
    return float(np.mean((velocity - target) ** 2))


def masking_demo(*, seed: int = 0, cfg: AudioImConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioImConfig()
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(128)
    xp, x1, m = split_latent_mask(latent, prompt_ratio=cfg.mask_prompt_ratio)
    t = 0.5
    xt = flow_interpolate(rng.standard_normal(128), x1, t)
    v = x1 - rng.standard_normal(128)
    return {
        "prompt_ratio": cfg.mask_prompt_ratio,
        "prompt_energy": float(np.sum(xp**2)),
        "target_energy": float(np.sum(x1**2)),
        "mask_sum": float(m.sum()),
        "flow_loss": flow_matching_loss(v, rng.standard_normal(128), x1),
        "xt_shape": list(xt.shape),
    }
