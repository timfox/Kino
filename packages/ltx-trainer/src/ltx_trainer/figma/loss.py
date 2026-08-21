"""Multi-view contrastive loss stubs (§3.1.2–3.1.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.figma.config import FigmaConfig


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_sim(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _l2_normalize(u) @ _l2_normalize(v).T


def global_contrastive_loss(
    z_audio: np.ndarray,
    z_text: np.ndarray,
    tau: float = 0.07,
) -> float:
    """Symmetric InfoNCE on global embeddings (§3.1.2)."""
    za = _l2_normalize(z_audio)
    zt = _l2_normalize(z_text)
    sim = za @ zt.T / tau
    b = sim.shape[0]
    labels = np.arange(b)
    loss_a2t = -np.mean(sim[np.arange(b), labels] - np.log(np.sum(np.exp(sim), axis=1)))
    loss_t2a = -np.mean(sim[labels, np.arange(b)] - np.log(np.sum(np.exp(sim.T), axis=1)))
    return float(0.5 * (loss_a2t + loss_t2a))


def frame_level_score(
    z_audio_frames: np.ndarray,
    z_text_tokens: np.ndarray,
) -> np.ndarray:
    """Eq. (1–2): max token sim per frame, averaged over frames."""
    # z_audio_frames: [B, T, D], z_text_tokens: [B, L, D]
    za = _l2_normalize(z_audio_frames)
    zt = _l2_normalize(z_text_tokens)
    # pairwise batch scores S(i,j)
    b, t, d = za.shape
    _, l, _ = zt.shape
    scores = np.zeros((b, b), dtype=np.float64)
    for i in range(b):
        for j in range(b):
            sim_tl = za[i] @ zt[j].T  # [T, L]
            scores[i, j] = np.mean(np.max(sim_tl, axis=1))
    return scores


def frame_contrastive_loss(
    z_audio_frames: np.ndarray,
    z_text_tokens: np.ndarray,
    tau: float = 0.07,
) -> float:
    """InfoNCE on frame-level scores (§3.1.3)."""
    s = frame_level_score(z_audio_frames, z_text_tokens) / tau
    b = s.shape[0]
    labels = np.arange(b)
    loss_a2t = -np.mean(s[np.arange(b), labels] - np.log(np.sum(np.exp(s), axis=1)))
    loss_t2a = -np.mean(s[labels, np.arange(b)] - np.log(np.sum(np.exp(s.T), axis=1)))
    return float(0.5 * (loss_a2t + loss_t2a))


def multi_view_loss(
    z_audio_global: np.ndarray,
    z_text_global: np.ndarray,
    z_audio_frames: np.ndarray,
    z_text_tokens: np.ndarray,
    *,
    alpha: float = 0.6,
    tau: float = 0.07,
) -> dict[str, float]:
    """L = α L_global + (1-α) L_frame (§3.1.4)."""
    lg = global_contrastive_loss(z_audio_global, z_text_global, tau=tau)
    lf = frame_contrastive_loss(z_audio_frames, z_text_tokens, tau=tau)
    total = alpha * lg + (1.0 - alpha) * lf
    return {"global": lg, "frame": lf, "total": float(total), "alpha": alpha}


def loss_demo(seed: int = 0, cfg: FigmaConfig | None = None) -> dict[str, Any]:
    c = cfg or FigmaConfig()
    rng = np.random.default_rng(seed)
    b, t, l, d = 8, 16, 12, c.embed_dim
    zg_a = rng.normal(size=(b, d))
    zg_t = zg_a + rng.normal(scale=0.3, size=(b, d))
    zf_a = rng.normal(size=(b, t, d))
    zf_t = rng.normal(size=(b, l, d))
    out = multi_view_loss(zg_a, zg_t, zf_a, zf_t, alpha=c.alpha_global, tau=c.temperature)
    out["trainable_params_m"] = c.trainable_params_m
    return out
