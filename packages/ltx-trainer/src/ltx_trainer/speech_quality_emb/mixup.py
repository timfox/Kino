"""Partial mix-up for frame-level pseudo targets (Eq. 3–4, arXiv:2605.21332)."""

from __future__ import annotations

import numpy as np


def partial_mixup_waveform_mask(
    t: int,
    *,
    rng: np.random.Generator,
    num_segments: tuple[int, int] = (1, 3),
    min_seg_len: int = 10,
    max_seg_len: int = 50,
) -> np.ndarray:
    """Binary mask m(t) ∈ {0,1}^T (stub): 1 selects degraded slice, 0 selects reference."""
    m = np.zeros(t, dtype=np.float64)
    nseg = int(rng.integers(num_segments[0], num_segments[1] + 1))
    for _ in range(nseg):
        seg_len = int(rng.integers(min_seg_len, min(max_seg_len, t) + 1))
        start = int(rng.integers(0, max(1, t - seg_len + 1)))
        m[start : start + seg_len] = 1.0
    return np.clip(m, 0.0, 1.0)


def pseudo_frame_scores(
    q_ref: np.ndarray,
    q_deg: np.ndarray,
    mq: np.ndarray,
) -> np.ndarray:
    """Eq. (4): q_pseudo = m_q ⊙ q̂_deg + (1 − m_q) ⊙ q̂_ref (toy uses provided score tracks)."""
    a = np.asarray(q_ref, dtype=np.float64)
    b = np.asarray(q_deg, dtype=np.float64)
    m = np.asarray(mq, dtype=np.float64)
    if a.shape != b.shape or a.shape != m.shape:
        raise ValueError("q_ref, q_deg, mq must share shape")
    return m * b + (1.0 - m) * a


def mix_waveforms_stub(
    s_ref: np.ndarray,
    s_deg: np.ndarray,
    m_time: np.ndarray,
) -> np.ndarray:
    """Eq. (3): s_pseudo(t) = m(t) s_deg + (1 − m(t)) s_ref (element-wise stub)."""
    a = np.asarray(s_ref, dtype=np.float64)
    b = np.asarray(s_deg, dtype=np.float64)
    m = np.asarray(m_time, dtype=np.float64)
    if a.shape != b.shape or a.shape != m.shape:
        raise ValueError("s_ref, s_deg, m_time must share shape")
    return m * b + (1.0 - m) * a
