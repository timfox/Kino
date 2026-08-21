"""Tsallis entropy confidence baseline — §3.1, Eqs. (1)–(2)."""

from __future__ import annotations

import numpy as np


def tsallis_token_confidence(probs: np.ndarray, alpha: float = 0.33) -> float:
    """C_token from vocabulary distribution — Eq. (1)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum()
    v = p.size
    sum_pa = float(np.sum(p**alpha))
    num = np.exp((v ** (1 - alpha) - sum_pa) / (1 - alpha)) - 1
    den = np.exp((v ** (1 - alpha) - 1) / (1 - alpha)) - 1
    return float(num / max(den, 1e-12))


def word_confidence(token_confs: list[float]) -> float:
    """Arithmetic mean over subword tokens — Eq. (2)."""
    if not token_confs:
        return 0.0
    return float(sum(token_confs) / len(token_confs))


def is_error_word(token_probs: list[np.ndarray], threshold: float) -> bool:
    confs = [tsallis_token_confidence(p) for p in token_probs]
    return word_confidence(confs) < threshold
