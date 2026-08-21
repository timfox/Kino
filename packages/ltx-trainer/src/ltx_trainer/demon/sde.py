"""SDE re-noise with per-frame source blending (DEMON §3.3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.demon.config import DemonConfig

PER_FRAME_CURVES: tuple[str, ...] = (
    "sde_denoise_curve",
    "guidance_curve",
    "velocity_scale",
    "ode_noise_curve",
    "apg_momentum",
    "cfg_rescale_curve",
    "x0_target_strength",
)


def sde_renoise_blend(
    x0_pred: np.ndarray,
    source_latents: np.ndarray,
    t_next: float,
    curve: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-frame blend between full SDE re-noise and source-anchored re-noise."""
    rng = rng or np.random.default_rng(0)
    noise = rng.standard_normal(x0_pred.shape)
    xt_full = t_next * noise + (1.0 - t_next) * x0_pred
    xt_source = t_next * noise + (1.0 - t_next) * source_latents
    c = np.asarray(curve, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    return c * xt_full + (1.0 - c) * xt_source


def ramp_curve(T: int, *, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return np.linspace(lo, hi, T, dtype=np.float64)


def segment_gradient(similarities: list[float]) -> float:
    """Segment 4 minus segment 1 mel similarity (Table 7/8 style)."""
    if len(similarities) < 4:
        return 0.0
    return similarities[3] - similarities[0]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def segment_source_similarities(
    blended: np.ndarray,
    source_latents: np.ndarray,
    *,
    n_segments: int = 4,
) -> list[float]:
    """Per-segment cosine similarity to source (computed, not anchored)."""
    T = blended.shape[0]
    seg_len = max(1, T // n_segments)
    sims: list[float] = []
    for i in range(n_segments):
        lo = i * seg_len
        hi = T if i == n_segments - 1 else (i + 1) * seg_len
        sims.append(_cosine_sim(blended[lo:hi].mean(axis=0), source_latents[lo:hi].mean(axis=0)))
    return sims


def sde_smoke(cfg: DemonConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or DemonConfig()
    rng = np.random.default_rng(seed)
    T, D = 16, min(32, cfg.latent_dim)
    x0 = rng.standard_normal((T, D))
    src = x0 + rng.normal(0, 0.1, (T, D))
    flat = sde_renoise_blend(x0, src, 0.3, np.ones(T), rng=rng)
    ramp = sde_renoise_blend(x0, src, 0.3, ramp_curve(T), rng=rng)
    sim_flat = segment_source_similarities(flat, src)
    sim_ramp = segment_source_similarities(ramp, src)
    return {
        "n_curves": len(PER_FRAME_CURVES),
        "flat_vs_ramp_diff": float(np.mean(np.abs(flat - ramp))),
        "ramp_gradient": segment_gradient(sim_ramp),
        "flat_gradient": segment_gradient(sim_flat),
        "latent_hz": cfg.latent_hz,
    }
