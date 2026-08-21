"""TTS RLHF generative reward shaping (toy)."""

from __future__ import annotations


def pairwise_grm_score(
    prompt: str,
    candidate_quality: float,
    reference_quality: float,
) -> float:
    """Toy relative GRM score in [-1, 1] from scalar quality proxies."""
    del prompt
    return float(np_clip(candidate_quality - reference_quality, -1.0, 1.0))


def reward_shaping(grm_score: float) -> float:
    """Map GRM relative score to policy reward (paper Eq. 1 uses transformation s(·))."""
    # Simple bounded shaping: emphasize preference margin while keeping signal smooth.
    return float(np_tanh(1.5 * grm_score))


def np_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def np_tanh(x: float) -> float:
    import math

    return math.tanh(x)
