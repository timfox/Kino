"""Dual-branch prompt adaptation stubs (Sec. 4.3, Eq. 9–12)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.entroad.config import EntroADConfig


def branch_prompt_bias(t_n: np.ndarray, t_a: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Eq. (9): lightweight MLP stub — independent params per branch via seed."""
    rng = np.random.default_rng(seed)
    pair = np.concatenate([np.asarray(t_n).reshape(-1), np.asarray(t_a).reshape(-1)])
    w1 = rng.standard_normal((pair.size, 32))
    w2 = rng.standard_normal((32, 16))
    h = np.tanh(pair @ w1)
    return (h @ w2).astype(np.float64)


def cosine_branch_maps(
    patch_features: np.ndarray,
    u_n: np.ndarray,
    u_a: np.ndarray,
    *,
    tau_s: float = 0.07,
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (11): softmax over normal vs anomaly cosine similarities."""
    z = np.asarray(patch_features, dtype=np.float64)
    un = np.asarray(u_n, dtype=np.float64).reshape(-1)
    ua = np.asarray(u_a, dtype=np.float64).reshape(-1)
    un = un / (np.linalg.norm(un) + 1e-8)
    ua = ua / (np.linalg.norm(ua) + 1e-8)
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    sn = zn @ un / tau_s
    sa = zn @ ua / tau_s
    stack = np.stack([sn, sa], axis=-1)
    stack = stack - stack.max(axis=-1, keepdims=True)
    ex = np.exp(stack)
    prob = ex / (ex.sum(axis=-1, keepdims=True) + 1e-8)
    return prob[..., 0], prob[..., 1]


def fuse_branch_maps(m_a: np.ndarray, m_b: np.ndarray, *, cfg: EntroADConfig | None = None) -> np.ndarray:
    """Eq. (20): M = ᾱ M_A + β̄ M_B."""
    cfg = cfg or EntroADConfig()
    a, b = cfg.fusion_alpha, cfg.fusion_beta
    denom = a + b + 1e-8
    return (a / denom) * np.asarray(m_a) + (b / denom) * np.asarray(m_b)


def image_score_from_map(
    anomaly_map: np.ndarray,
    retrieval_score: float,
    *,
    cfg: EntroADConfig | None = None,
    top_frac: float = 0.01,
) -> float:
    """Eq. (21–22): top-1% mean + retrieval blend."""
    cfg = cfg or EntroADConfig()
    m = np.asarray(anomaly_map, dtype=np.float64).reshape(-1)
    k = max(1, int(len(m) * top_frac))
    top = np.partition(m, -k)[-k:]
    a_loc = float(top.mean())
    return float((1.0 - cfg.inference_k) * a_loc + cfg.inference_k * retrieval_score)
