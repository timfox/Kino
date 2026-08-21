"""Second-stage phoneme matcher (toy) for DMA-KWS (arXiv:2605.22120).

The paper's second stage uses a QbyT phoneme matcher with utterance-level and phoneme-level losses.
Here we implement a lightweight similarity verifier used as a stub for Score2.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def phoneme_match_score(audio_embed: np.ndarray, phoneme_proto: np.ndarray) -> float:
    """Toy utterance-level score in [0, 1] derived from cosine similarity."""
    sim = cosine_similarity(audio_embed, phoneme_proto)
    # map [-1, 1] → [0, 1]
    return float(0.5 * (sim + 1.0))


def two_stage_score(score1: float, score2: float, alpha: float = 0.5) -> float:
    """Combine Stage-1 and Stage-2 scores (not in paper; stub convenience)."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * float(score1) + (1.0 - a) * float(score2)

