"""FiLM speaker conditioning stubs (arXiv:2606.06211)."""

from __future__ import annotations

import numpy as np


def mask_speaker_embedding(z: np.ndarray, is_normative: bool) -> np.ndarray:
    """§2.1 Eq. (2): zero x-vector for healthy/normative speech."""
    if z.ndim != 1:
        raise ValueError("z must be 1-D speaker embedding")
    if is_normative:
        return np.zeros_like(z)
    return z


def film_generator(z: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """§2.2 Eq. (3–4): MLP maps speaker embedding to (gamma, beta)."""
    h = np.maximum(z @ w1.T + b1, 0.0)
    out = h @ w2.T + b2
    d = out.shape[0] // 2
    return out[:d], out[d:]


def gate_alpha(z: np.ndarray, wg: np.ndarray, bg: float | np.ndarray) -> float:
    """§2.2 Eq. (5): alpha_l = sigma(G_g(z))."""
    bias = float(np.asarray(bg).reshape(-1)[0])
    logit = float(z @ wg + bias)
    return 1.0 / (1.0 + np.exp(-logit))


def gated_film_modulate(
    hidden: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """§2.2 Eq. (6): eH = H + alpha * ((gamma-1) * H + beta)."""
    if hidden.ndim != 2:
        raise ValueError("hidden must be (seq, dim)")
    mod = (gamma - 1.0) * hidden + beta
    return hidden + alpha * mod


def identity_init_gamma_beta(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """§3.5: gamma=1, beta=0 at initialization."""
    return np.ones(dim), np.zeros(dim)
