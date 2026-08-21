"""RAHT prelude + transform (Sec. 3, Eq. 1)."""

from __future__ import annotations

import numpy as np


def raht_prelude(morton_codes: np.ndarray, *, depth: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build (I_l, W_l, F_l) schedule for 3J binary levels (Alg. A.4)."""
    n = len(morton_codes)
    m = morton_codes.astype(np.int64)
    active = np.arange(n, dtype=np.int64)
    schedule: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    levels = 3 * depth
    for ell in range(1, levels + 1):
        idx = active
        m_ell = m[idx]
        m_len = len(idx)
        if m_len == 0:
            break
        weights = np.empty(m_len, dtype=np.float64)
        flags = np.zeros(m_len, dtype=bool)
        mask = (1 << (3 * depth)) - (1 << ell)
        for k in range(m_len):
            curr = idx[k]
            nxt = idx[k + 1] if k + 1 < m_len else n
            weights[k] = float(nxt - curr)
            if k + 1 < m_len:
                diff = int(m_ell[k]) ^ int(m_ell[k + 1])
                flags[k] = (diff & mask) == 0
        schedule.append((idx.copy(), weights, flags))
        if m_len <= 1:
            break
        keep = np.ones(m_len, dtype=bool)
        if m_len > 1:
            keep[1:] = ~flags[:-1]
        active = idx[keep]
    return schedule


def raht_forward(values: np.ndarray, schedule: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Apply sibling Haar-like transform along attribute vector."""
    y = values.astype(np.float64).copy()
    for indices, weights, flags in schedule:
        for k in range(len(indices) - 1):
            if not flags[k]:
                continue
            i0, i1 = int(indices[k]), int(indices[k + 1])
            w0, w1 = weights[k], weights[k + 1]
            denom = np.sqrt(w0 + w1) + 1e-8
            a, b = np.sqrt(w0) / denom, np.sqrt(w1) / denom
            y0, y1 = y[i0], y[i1]
            y[i0] = a * y0 + b * y1
            y[i1] = -b * y0 + a * y1
    return y


def raht_inverse(coeffs: np.ndarray, schedule: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    y = coeffs.astype(np.float64).copy()
    for indices, weights, flags in reversed(schedule):
        for k in range(len(indices) - 1):
            if not flags[k]:
                continue
            i0, i1 = int(indices[k]), int(indices[k + 1])
            w0, w1 = weights[k], weights[k + 1]
            denom = np.sqrt(w0 + w1) + 1e-8
            a, b = np.sqrt(w0) / denom, np.sqrt(w1) / denom
            y0, y1 = y[i0], y[i1]
            y[i0] = a * y0 - b * y1
            y[i1] = b * y0 + a * y1
    return y


def high_frequency_energy(coeffs: np.ndarray, schedule: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """Energy in high-frequency sibling slots (should drop after RAHT)."""
    energy = 0.0
    for indices, _, flags in schedule:
        for k in range(len(indices) - 1):
            if flags[k]:
                energy += float(coeffs[int(indices[k + 1])] ** 2)
    return energy
