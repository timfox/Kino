"""Representation sensitivity under controlled perturbations (Sec. 3.4)."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def expected_random_similarity(embeddings: np.ndarray, *, pairs: int = 1000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    idx_a = rng.integers(0, n, size=pairs)
    idx_b = rng.integers(0, n, size=pairs)
    sims = [cosine_similarity(embeddings[i], embeddings[j]) for i, j in zip(idx_a, idx_b, strict=True)]
    return float(np.mean(sims))


def sensitivity_delta(reference: np.ndarray, perturbed: np.ndarray, mu: float) -> float:
    """Δ(x, x′) = 1 − (s − μ) / (1 − μ)."""
    s = cosine_similarity(reference, perturbed)
    denom = 1.0 - mu
    if abs(denom) < 1e-8:
        return 0.0
    return 1.0 - (s - mu) / denom


def batch_sensitivity(
    references: np.ndarray,
    perturbed: np.ndarray,
    *,
    mu: float | None = None,
    seed: int = 0,
) -> float:
    if mu is None:
        mu = expected_random_similarity(references, seed=seed)
    deltas = [sensitivity_delta(r, p, mu) for r, p in zip(references, perturbed, strict=True)]
    return float(np.mean(deltas))
