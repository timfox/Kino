"""Normalized AE bottleneck, RQ-MTP, and patch-flow stubs (arXiv:2606.06357)."""

from __future__ import annotations

import numpy as np


def channel_normalize(z0: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """§3.1 Eq. (1): per-time-step channel normalization."""
    if z0.ndim != 2:
        raise ValueError("z0 must be (time, channels)")
    mu = z0.mean(axis=-1, keepdims=True)
    sigma = z0.std(axis=-1, keepdims=True)
    return (z0 - mu) / (sigma + eps)


def noise_perturb(
    zn: np.ndarray,
    *,
    gamma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """§3.1 Eq. (2): z_tilde = z_n + alpha * eps, alpha ~ U(0, gamma)."""
    alpha = float(rng.uniform(0.0, gamma))
    noise = rng.normal(0.0, 1.0, size=zn.shape)
    return zn + alpha * noise, alpha


def rq_mtp_loss(logits_list: list[np.ndarray], targets: list[np.ndarray]) -> float:
    """§3.2 Eq. (5): sum of CE over K RQ-MTP heads."""
    eps = 1e-9
    total = 0.0
    for logits, target in zip(logits_list, targets, strict=True):
        p = np.clip(logits, eps, 1.0)
        p /= p.sum(axis=-1, keepdims=True)
        total += float(-np.mean(np.sum(target * np.log(p), axis=-1)))
    return total / max(len(logits_list), 1)


def llm_ce_loss(log_probs: np.ndarray, targets: np.ndarray) -> float:
    """§3.2 Eq. (7): frozen-LLM cross-entropy stub."""
    eps = 1e-9
    p = np.clip(log_probs, eps, 1.0)
    return float(-np.mean(np.sum(targets * np.log(p), axis=-1)))


def patch_targets(z: np.ndarray, patch_frames: int) -> list[np.ndarray]:
    """§3.3 Eq. (9): contiguous z patches of length P."""
    if z.ndim != 2:
        raise ValueError("z must be (time, channels)")
    if patch_frames <= 0 or z.shape[0] < patch_frames:
        raise ValueError("invalid patch size for latent sequence")
    patches: list[np.ndarray] = []
    for start in range(0, z.shape[0] - patch_frames + 1, patch_frames):
        patches.append(z[start : start + patch_frames])
    return patches


def flow_matching_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """§3.3: patch-level flow-matching objective stub."""
    return float(np.mean((pred - target) ** 2))
