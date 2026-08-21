"""Toy representation-geometry drift metrics."""

from __future__ import annotations

import numpy as np


def _class_separability(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Between-class / within-class variance ratio (higher = better separated)."""
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 1.0
    global_mean = embeddings.mean(axis=0)
    between = 0.0
    within = 0.0
    for lab in unique:
        mask = labels == lab
        cls = embeddings[mask]
        if cls.shape[0] < 2:
            continue
        mu = cls.mean(axis=0)
        between += mask.sum() * float(np.sum((mu - global_mean) ** 2))
        within += float(np.sum((cls - mu) ** 2))
    return between / max(within, 1e-9)


def simulate_entangled_embeddings(
    n_per_class: int,
    rng: np.random.Generator,
    dim: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Two factors (phonetic + speaker) entangled in shared 2D subspace per class."""
    labels = np.repeat(np.arange(3), n_per_class)
    base = rng.standard_normal((3, dim))
    embeddings = []
    for i, lab in enumerate(labels):
        noise = 0.15 * rng.standard_normal(dim)
        speaker_shift = 0.4 * rng.standard_normal(dim) * (i % 2)
        embeddings.append(base[lab] + speaker_shift + noise)
    return np.stack(embeddings), labels


def apply_adaptation_drift(
    embeddings: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Collapse toward global mean — toy catastrophic geometry drift."""
    global_mean = embeddings.mean(axis=0)
    noise = 0.05 * strength * rng.standard_normal(embeddings.shape)
    return embeddings + strength * (global_mean - embeddings) + noise


def drift_report(
    n_per_class: int = 40,
    seed: int = 0,
    drift_strength: float = 0.8,
) -> dict[str, float]:
    """Compare separability before/after toy adaptation."""
    rng = np.random.default_rng(seed)
    emb, labels = simulate_entangled_embeddings(n_per_class, rng)
    sep_before = _class_separability(emb, labels)
    emb_after = apply_adaptation_drift(emb, drift_strength, rng)
    sep_after = _class_separability(emb_after, labels)
    return {
        "phonetic_separability_before": sep_before,
        "phonetic_separability_after": sep_after,
        "relative_drop": (sep_before - sep_after) / max(sep_before, 1e-9),
        "drift_strength": drift_strength,
    }


def peft_style_update(
    embeddings: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    rank: int = 2,
) -> np.ndarray:
    """Low-rank perturbation — weaker drift than full mean-collapse."""
    dim = embeddings.shape[1]
    u = rng.standard_normal((dim, rank)) * strength * 0.15
    v = rng.standard_normal((rank, dim)) * strength * 0.15
    delta = u @ v
    global_mean = embeddings.mean(axis=0)
    return embeddings + embeddings @ delta + 0.25 * strength * (global_mean - embeddings)
