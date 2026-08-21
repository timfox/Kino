"""Conditional consistency loss — §3.2.3, Eq. (4)."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def prosody_cross_entropy(
    logits: np.ndarray,
    target_cp: np.ndarray,
) -> float:
    """Softmax CE for discrete prosody token prediction."""
    logits = np.asarray(logits, dtype=np.float64)
    target = int(np.argmax(target_cp)) % logits.shape[0]
    log_z = logits - np.max(logits)
    probs = np.exp(log_z)
    probs /= probs.sum() + 1e-12
    return float(-np.log(probs[target] + 1e-12))


def conditional_consistency_loss(
    x_hat: np.ndarray,
    cp: np.ndarray,
    z_spk: np.ndarray,
    prosody_logits: np.ndarray,
    spk_pred: np.ndarray,
    lambda_pro: float = 0.2,
    lambda_spk: float = 0.5,
) -> dict[str, float]:
    """LCCL = λ_pro CE(cp, f(x̂, z_spk)) - λ_spk cos(z_spk, g(x̂, cp)) — Eq. (4)."""
    ce = prosody_cross_entropy(prosody_logits, cp)
    cos = cosine_similarity(z_spk, spk_pred)
    total = lambda_pro * ce - lambda_spk * cos
    return {
        "L_ccl": total,
        "L_ce_prosody": ce,
        "cos_spk": cos,
        "lambda_pro": lambda_pro,
        "lambda_spk": lambda_spk,
    }
