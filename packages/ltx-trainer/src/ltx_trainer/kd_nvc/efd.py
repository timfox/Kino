"""Energy-aware feature distillation (EFD) — Eq. (5–7)."""

from __future__ import annotations

import numpy as np


def channel_energy(feature: np.ndarray) -> np.ndarray:
    """Per-channel mean absolute response e_c (Section I footnote)."""
    return np.abs(feature).mean(axis=(1, 2))


def top10_energy_ratio(energies: np.ndarray) -> float:
    """T10: fraction of total energy in top-10% channels."""
    if energies.size == 0:
        return 0.0
    n_top = max(1, int(np.ceil(0.1 * energies.size)))
    order = np.argsort(energies)[::-1]
    top = energies[order[:n_top]].sum()
    total = energies.sum() + 1e-12
    return float(top / total)


def normalized_gini(energies: np.ndarray) -> float:
    """Normalized Gini over channel energies (Fig. 2c)."""
    x = np.sort(np.maximum(energies, 0.0))
    n = x.size
    if n == 0:
        return 0.0
    cum = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-12)) / n
    return float(np.clip(g, 0.0, 1.0))


def adaptive_energy_signature(feature: np.ndarray, *, k: int = 8) -> np.ndarray:
    """Spatially aggregated energy signature E = AdaptiveAvgPool_k×k(F) (Eq. 5)."""
    c, h, w = feature.shape
    kh = max(1, min(k, h))
    kw = max(1, min(k, w))
    out_h = max(1, h // kh)
    out_w = max(1, w // kw)
    sig = np.zeros((c, out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            patch = feature[:, i * kh : (i + 1) * kh, j * kw : (j + 1) * kw]
            sig[:, i, j] = np.abs(patch).mean()
    return sig


def efd_loss(
    teacher: np.ndarray,
    student: np.ndarray,
    *,
    k: int = 8,
) -> float:
    """LEFD from Eq. (6)."""
    et = adaptive_energy_signature(teacher, k=k)
    es = adaptive_energy_signature(student, k=k)
    if es.shape != et.shape:
        kh, kw = et.shape[1], et.shape[2]
        es = adaptive_energy_signature(student, k=min(k, student.shape[1], student.shape[2]))
        if es.shape[1] != kh or es.shape[2] != kw:
            es = es[:, :kh, :kw] if es.shape[1] >= kh else np.pad(es, ((0, 0), (0, kh - es.shape[1]), (0, 0)))
    diff = et - es
    c, kh, kw = et.shape
    return float(np.mean(diff**2))


def total_loss(
    rd_loss: float,
    teacher_feats: list[np.ndarray],
    student_feats: list[np.ndarray],
    *,
    beta: float = 1.0,
    k: int = 8,
) -> float:
    """Ltotal = LRD + β Σ LEFD (Eq. 7)."""
    efd = sum(efd_loss(t, s, k=k) for t, s in zip(teacher_feats, student_feats, strict=True))
    return rd_loss + beta * efd
