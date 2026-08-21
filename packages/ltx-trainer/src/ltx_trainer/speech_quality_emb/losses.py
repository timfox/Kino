"""Toy LSSQA + pseudo-supervision + supervised contrastive terms (Eq. 5–8, arXiv:2605.21332)."""

from __future__ import annotations

import numpy as np


def utterance_mae_loss(y_hat_frames: np.ndarray, y_utterance: float) -> float:
    """Surrogate utterance loss: mean absolute error of frame preds to scalar target."""
    q = np.asarray(y_hat_frames, dtype=np.float64).ravel()
    y = float(y_utterance)
    return float(np.mean(np.abs(q - y)))


def frame_pseudo_l1(q_hat: np.ndarray, q_pseudo: np.ndarray) -> float:
    """Eq. (5) term: mean |q̂ − q_pseudo| over frames (scalar stub)."""
    a = np.asarray(q_hat, dtype=np.float64).ravel()
    b = np.asarray(q_pseudo, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("q_hat and q_pseudo must match shape")
    return float(np.mean(np.abs(a - b)))


def supervised_contrastive_stub(
    z_proj: np.ndarray,
    labels: np.ndarray,
    *,
    temperature: float = 0.1,
) -> float:
    """Toy scalar contrastive stress: encourage same-class cosine similarity > cross-class mean.

    Full Eq. (6–7) is batch softmax over all frame pairs; this stub captures the sign of the gradient
    for agent orientation without O((BL)^2) allocations.
    """
    z = np.asarray(z_proj, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).ravel()
    if z.ndim != 2:
        raise ValueError("z_proj must be (N, D)")
    if z.shape[0] != y.size:
        raise ValueError("labels length must match number of rows")
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    tau = max(float(temperature), 1e-6)

    pos_pull = 0.0
    neg_push = 0.0
    n_pos = 0
    n_neg = 0
    for i in range(z.shape[0]):
        sim = z @ z[i]
        same = y == y[i]
        diff = ~same
        # exclude self
        sim_i = sim.copy()
        sim_i[i] = -np.inf
        if np.any(same & (np.arange(z.shape[0]) != i)):
            pos_pull += float(np.max(sim_i[same]))
            n_pos += 1
        if np.any(diff):
            neg_push += float(np.mean(sim_i[diff]))
            n_neg += 1

    pos_term = pos_pull / max(n_pos, 1)
    neg_term = neg_push / max(n_neg, 1)
    # minimize: negative log-like margin (toy)
    return float(-np.log((np.exp(pos_term / tau) + 1e-8) / (np.exp(neg_term / tau) + np.exp(pos_term / tau) + 1e-8)))


def total_training_loss(
    l_lssqa: float,
    l_frame_sup: float,
    l_scl: float,
    *,
    tau: float = 0.1,
    alpha_frame: float = 1.0,
) -> float:
    """Eq. (8): L_total = L^sup_LSSQA + τ · L_scl (frame pseudo term folded into L^sup stub here as additive)."""
    return float(l_lssqa + alpha_frame * l_frame_sup + float(tau) * l_scl)
