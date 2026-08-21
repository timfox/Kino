"""Post-hoc Russell Circumplex projection for emotion2vec (arXiv:2605.22732, Table 1)."""

from __future__ import annotations

import numpy as np

# emotion2vec_plus_large classes → (wA, wV) from paper Table 1
E2V_RUSSELL_WEIGHTS: dict[str, tuple[float, float]] = {
    "angry": (0.75, -0.75),
    "disgusted": (0.60, -0.80),
    "fearful": (0.80, -0.65),
    "happy": (0.65, 0.90),
    "neutral": (0.00, 0.00),
    "other": (0.10, 0.00),
    "sad": (-0.30, -0.85),
    "surprised": (0.70, 0.20),
}

E2V_CLASSES = tuple(E2V_RUSSELL_WEIGHTS.keys())


def russell_arousal_valence(class_probs: dict[str, float] | np.ndarray) -> tuple[float, float]:
    """Eq. (1–2): weighted sum of class probabilities."""
    if isinstance(class_probs, np.ndarray):
        probs = {c: float(class_probs[i]) for i, c in enumerate(E2V_CLASSES)}
    else:
        probs = {k: float(v) for k, v in class_probs.items()}
    arousal = 0.0
    valence = 0.0
    for cls, (wa, wv) in E2V_RUSSELL_WEIGHTS.items():
        p = probs.get(cls, 0.0)
        arousal += p * wa
        valence += p * wv
    return arousal, valence
