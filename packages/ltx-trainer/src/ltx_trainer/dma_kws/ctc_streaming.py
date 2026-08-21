"""CTC streaming phoneme search (toy) for DMA-KWS (arXiv:2605.22120).

Implements Algorithm 1 style max-product forward recursion on a blank-inserted target.
This is a *paper-stub* implementation: it operates on already-normalized posteriors.
"""

from __future__ import annotations

import numpy as np


BLANK_ID = 0


def blank_insert(target: list[int], blank_id: int = BLANK_ID) -> list[int]:
    """Insert blanks between labels and at ends: [ϕ, w1, ϕ, ..., wU, ϕ]."""
    out: list[int] = [blank_id]
    for t in target:
        out.append(int(t))
        out.append(blank_id)
    return out


def ctc_streaming_score(
    posteriors: np.ndarray,
    target: list[int],
    blank_id: int = BLANK_ID,
) -> np.ndarray:
    """Return frame-level score sequence (length T) for target phoneme sequence.

    posteriors: shape (T, V), each row a distribution over labels (incl blank).
    target: list of label ids (excluding blanks).

    Uses a max-product recursion mirroring the paper's Algorithm 1.
    """
    p = np.asarray(posteriors, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("posteriors must be 2D (T, V)")
    t_len, vocab = p.shape
    if t_len == 0:
        return np.zeros((0,), dtype=np.float64)
    if not target:
        # empty keyword: treat as always-on (not meaningful), return zeros
        return np.zeros((t_len,), dtype=np.float64)

    w = blank_insert(target, blank_id=blank_id)
    u_len = len(w)

    # δ(t, u) stored for current t only (streaming)
    prev = np.zeros((u_len,), dtype=np.float64)
    curr = np.zeros((u_len,), dtype=np.float64)

    # paper init: δ(1,1)=δ(1,2)=1 then multiply by p1(...)
    # Here we use 0-indexed: positions 0 and 1 correspond to u=1 and u=2.
    prev[:] = 0.0
    prev[0] = 1.0
    if u_len > 1:
        prev[1] = 1.0

    # incorporate first frame emission
    prev[0] *= p[0, blank_id]
    if u_len > 1:
        prev[1] *= p[0, w[1]]

    scores = np.zeros((t_len,), dtype=np.float64)
    scores[0] = float(max(prev[u_len - 2], prev[u_len - 1])) if u_len >= 2 else float(prev[0])

    for t in range(1, t_len):
        curr[:] = 0.0
        for u in range(u_len):
            label = w[u]
            if label == blank_id:
                stay = prev[u]
                prev_u1 = prev[u - 1] if u - 1 >= 0 else 0.0
                curr[u] = p[t, blank_id] * max(stay, prev_u1)
            else:
                stay = prev[u]
                prev_u1 = prev[u - 1] if u - 1 >= 0 else 0.0
                prev_u2 = prev[u - 2] if u - 2 >= 0 else 0.0
                curr[u] = p[t, label] * max(stay, prev_u1, prev_u2)
        scores[t] = float(max(curr[u_len - 2], curr[u_len - 1])) if u_len >= 2 else float(curr[0])
        prev, curr = curr, prev

    return scores


def find_candidate_segments(scores: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Find contiguous segments where score exceeds threshold (toy timestamping)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return []
    above = s >= float(threshold)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, ok in enumerate(above.tolist()):
        if ok and start is None:
            start = i
        if (not ok) and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, int(s.size)))
    return segments

