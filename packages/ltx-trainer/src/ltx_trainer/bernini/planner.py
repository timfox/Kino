"""MLLM semantic planner — masked refinement schedule (Sec. 2.1.1)."""

from __future__ import annotations

import math


def inference_mask_ratio(step_k: int, total_K: int) -> float:
    r"""mask_ratio(k, K) = cos(π/2 · (k+1)/K) — fraction of target tokens still masked (toy)."""
    if total_K <= 0:
        return 0.0
    k = max(0, min(step_k, total_K - 1))
    return math.cos(0.5 * math.pi * (k + 1) / total_K)


def train_mask_ratio_beta(*, alpha: float = 5.0, beta: float = 1.1) -> float:
    """Sample r ~ Beta(α, β) for training mask fraction (Sec. 4.1, Table 3 style)."""
    # Deterministic stub: mean alpha/(alpha+beta) jittered for smoke
    mean = alpha / (alpha + beta)
    return max(0.0, min(1.0, mean + 0.05 * math.sin(alpha + beta)))


def visible_token_fraction(mask_ratio: float) -> float:
    """Complement of masked fraction."""
    return max(0.0, min(1.0, 1.0 - mask_ratio))
