"""Flow-matching DiT velocity predictor (Dasheng AudioGen §3.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.dasheng_audiogen.config import DashengAudioGenConfig


@dataclass
class FlowMatchingDiT:
    """Lightweight cross-attention velocity field v_theta(z_t, t, text)."""

    latent_dim: int
    text_dim: int = 256
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        rng = self.rng or np.random.default_rng(0)
        d, td = self.latent_dim, self.text_dim
        self.w_q = rng.standard_normal((d, d)) * 0.02
        self.w_k = rng.standard_normal((td, d)) * 0.02
        self.w_v = rng.standard_normal((td, d)) * 0.02
        self.w_out = rng.standard_normal((d, d)) * 0.02
        self.time_emb = rng.standard_normal((1, d)) * 0.01

    def predict_velocity(self, z_t: np.ndarray, t: float, text_emb: np.ndarray) -> np.ndarray:
        z = np.asarray(z_t, dtype=np.float64)
        te = np.asarray(text_emb, dtype=np.float64)
        if te.ndim == 1:
            te = te[None, :]
        q = z @ self.w_q
        k = te @ self.w_k
        v = te @ self.w_v
        attn = np.tanh(q @ k.T / np.sqrt(q.shape[-1]))
        ctx = attn @ v
        t_scale = 1.0 - t
        return ctx @ self.w_out + self.time_emb * t_scale


def predict_velocity(
    z_t: np.ndarray,
    t: float,
    text_emb: np.ndarray,
    *,
    cfg: DashengAudioGenConfig | None = None,
    model: FlowMatchingDiT | None = None,
) -> np.ndarray:
    cfg = cfg or DashengAudioGenConfig()
    m = model or FlowMatchingDiT(latent_dim=z_t.shape[-1] if z_t.ndim > 1 else cfg.latent_dim)
    return m.predict_velocity(z_t, t, text_emb)


def model_smoke(cfg: DashengAudioGenConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or DashengAudioGenConfig()
    rng = np.random.default_rng(seed)
    T = int(cfg.latent_hz * 2.0)
    d = min(64, cfg.latent_dim)
    z = rng.standard_normal((T, d))
    text = rng.standard_normal((cfg.text_dim,))
    m = FlowMatchingDiT(latent_dim=d, text_dim=cfg.text_dim, rng=rng)
    v = m.predict_velocity(z, 0.5, text)
    return {"velocity_shape": list(v.shape), "finite": bool(np.all(np.isfinite(v)))}
