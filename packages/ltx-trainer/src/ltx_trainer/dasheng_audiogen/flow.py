"""Flow-matching objective stub (Dasheng AudioGen §3.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dasheng_audiogen.config import DashengAudioGenConfig
from ltx_trainer.dasheng_audiogen.model import FlowMatchingDiT


def flow_matching_loss(
    v_pred: np.ndarray,
    z0: np.ndarray,
    z1: np.ndarray,
) -> float:
    """L_FM = ||v_theta - (z1 - z0)||^2."""
    target = z1 - z0
    return float(np.mean((v_pred - target) ** 2))


def linear_interpolate(z0: np.ndarray, z1: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * z0 + t * z1


def flow_smoke(cfg: DashengAudioGenConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or DashengAudioGenConfig()
    rng = np.random.default_rng(seed)
    T = int(cfg.latent_hz * cfg.clip_duration_s)
    d = min(32, cfg.latent_dim)  # toy dim for smoke
    z0 = rng.standard_normal((T, d))
    z1 = rng.standard_normal((T, d))
    t = 0.5
    zt = linear_interpolate(z0, z1, t)
    v_target = z1 - z0
    text = rng.standard_normal((cfg.text_dim,))
    model = FlowMatchingDiT(latent_dim=d, text_dim=cfg.text_dim, rng=rng)
    v_pred = model.predict_velocity(zt, t, text)
    loss = flow_matching_loss(v_pred, z0, z1)
    return {
        "latent_frames": T,
        "latent_dim_full": cfg.latent_dim,
        "fm_steps_inference": cfg.fm_steps,
        "loss_finite": np.isfinite(loss),
        "uses_dit_velocity": True,
    }
