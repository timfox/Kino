"""Semantic latent Chain-of-Thought (Eq. 1–6) stubs."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.planaudio.config import PlanAudioConfig


def latent_mse_loss(z: np.ndarray, h: np.ndarray) -> float:
    """MSE term in Eq. (5)."""
    z = np.asarray(z, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    return float(np.mean((z - h) ** 2))


def latent_cosine_penalty(z: np.ndarray, h: np.ndarray) -> float:
    """1 - cosine similarity term in Eq. (5)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(z) * np.linalg.norm(h) + 1e-12
    cos = float(np.dot(z, h) / denom)
    return 1.0 - cos


def latent_supervision_loss(
    z: np.ndarray,
    h: np.ndarray,
    *,
    cfg: PlanAudioConfig | None = None,
) -> dict[str, float]:
    """Eq. (5): MSE + λ * (1 - cos)."""
    cfg = cfg or PlanAudioConfig()
    mse = latent_mse_loss(z, h)
    cos_pen = latent_cosine_penalty(z, h)
    total = mse + cfg.loss_lambda_cosine * cos_pen
    return {"mse": mse, "cosine_penalty": cos_pen, "L_latent": total}


def audio_ce_loss(log_probs: list[float]) -> float:
    """Negative log-likelihood stub for hierarchical tokens (Eq. 6)."""
    if not log_probs:
        return float("nan")
    return float(-sum(log_probs) / len(log_probs))


def total_loss(
    latent_loss: float,
    audio_loss: float,
    *,
    cfg: PlanAudioConfig | None = None,
) -> float:
    cfg = cfg or PlanAudioConfig()
    return cfg.loss_lambda_latent * latent_loss + cfg.loss_lambda_audio * audio_loss


def format_sequence(
    text_tokens: list[str],
    *,
    latent_dim: int = 8,
    audio_token_count: int = 4,
    cfg: PlanAudioConfig | None = None,
) -> dict[str, Any]:
    """Eq. (2) sequence layout <|sot|> x <|sol|> z <|soa|> y <|eoa|>."""
    cfg = cfg or PlanAudioConfig()
    z = np.random.default_rng(42).standard_normal((cfg.latent_cot_steps, latent_dim))
    y = [f"a{i}" for i in range(audio_token_count)]
    return {
        "tokens": [
            cfg.special_tokens[0],
            *text_tokens,
            cfg.special_tokens[1],
            f"z[{cfg.latent_cot_steps}x{latent_dim}]",
            cfg.special_tokens[2],
            *y,
            cfg.special_tokens[3],
        ],
        "latent_shape": list(z.shape),
        "factorization": "P(y|x) = P(z|x) P(y|x,z)",
    }


def downsample_af3_embeddings(h: np.ndarray, *, cfg: PlanAudioConfig | None = None) -> np.ndarray:
    """Pool 750 AF3 steps → K=6 semantic plan (Sec. 5.1)."""
    cfg = cfg or PlanAudioConfig()
    h = np.asarray(h, dtype=np.float64)
    if h.ndim == 1:
        h = h.reshape(-1, 1)
    n, d = h.shape
    stride = max(1, n // cfg.latent_cot_steps)
    chunks = [h[i : i + stride].mean(axis=0) for i in range(0, n, stride)][: cfg.latent_cot_steps]
    while len(chunks) < cfg.latent_cot_steps:
        chunks.append(chunks[-1] if chunks else np.zeros(d))
    return np.stack(chunks, axis=0)
