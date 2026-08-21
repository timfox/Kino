"""ΔFLAM and paper evaluation metrics (stubs)."""

from __future__ import annotations

import numpy as np


def p_flam_max_frame(prompt: str, audio_features: np.ndarray, *, seed: int = 0) -> float:
    """
    Toy FLAM max frame probability P_FLAM(c, A).

    In the paper, FLAM returns frame-level event scores; here we use a deterministic
    hash of prompt vs mean feature energy as a stand-in.
    """
    rng = np.random.default_rng(hash((prompt, seed)) % (2**32))
    base = float(np.mean(np.abs(audio_features)))
    bias = (hash(prompt) % 1000) / 1000.0
    noise = rng.uniform(0.0, 0.05)
    return float(np.clip(0.3 * base + 0.5 * bias + noise, 0.0, 1.0))


def delta_flam(
    p_target: float,
    p_source: float,
) -> float:
    """ΔFLAM = P_FLAM(c_tar, A) − P_FLAM(c_src, A)."""
    return float(p_target - p_source)


def positive_delta_flam_ratio(deltas: list[float]) -> float:
    """Fraction of clips with ΔFLAM > 0 (paper positive-ratio)."""
    if not deltas:
        return 0.0
    return float(sum(1 for d in deltas if d > 0) / len(deltas))
