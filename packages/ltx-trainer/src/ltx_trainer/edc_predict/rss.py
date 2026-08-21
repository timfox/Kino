"""Random Sign-Sticky (RSS) toy for polarity assignment (paper Sec. RIR reconstruction)."""

from __future__ import annotations

import numpy as np


def random_sign_sticky_sequence(
    length: int,
    *,
    stickiness_p: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate ±1 sequence: with probability p, repeat previous sign; else flip.

    This is a minimal stand-in for RSS (paper: p = 0.9) — not a full RIR synthesizer.
    """
    if length <= 0:
        return np.array([], dtype=np.int8)
    g = rng or np.random.default_rng(0)
    p = float(np.clip(stickiness_p, 0.0, 1.0))
    out = np.empty(length, dtype=np.int8)
    s = 1 if g.random() < 0.5 else -1
    out[0] = s
    for i in range(1, length):
        if g.random() < p:
            out[i] = out[i - 1]
        else:
            out[i] = -int(out[i - 1])
    return out
