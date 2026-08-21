"""Multi-verifier proxies: VideoReward-TA + JavisScore (arXiv:2606.03183)."""

from __future__ import annotations

from typing import Any

import numpy as np


def video_reward_ta_proxy(
    *,
    text_overlap: float,
    motion_stability: float,
    seed: int = 0,
) -> float:
    """Toy text-alignment verifier in [0, 1]."""
    rng = np.random.default_rng(seed)
    base = 0.55 * float(np.clip(text_overlap, 0.0, 1.0)) + 0.45 * float(np.clip(motion_stability, 0.0, 1.0))
    return float(np.clip(base + rng.normal(0, 0.02), 0.0, 1.0))


def javis_score_proxy(
    *,
    av_sync: float,
    fine_grained_match: float,
    seed: int = 0,
) -> float:
    """Toy AV sync verifier in [0, 1]."""
    rng = np.random.default_rng(seed + 1)
    base = 0.6 * float(np.clip(av_sync, 0.0, 1.0)) + 0.4 * float(np.clip(fine_grained_match, 0.0, 1.0))
    return float(np.clip(base + rng.normal(0, 0.02), 0.0, 1.0))


def combined_score(
    vr: float,
    js: float,
    *,
    vr_weight: float = 0.5,
    javis_weight: float = 0.5,
) -> float:
    w_sum = max(vr_weight + javis_weight, 1e-9)
    return (vr_weight * vr + javis_weight * js) / w_sum


def score_candidate(candidate: dict[str, Any], *, seed: int, vr_weight: float, javis_weight: float) -> dict[str, float]:
    vr = video_reward_ta_proxy(
        text_overlap=float(candidate.get("text_overlap", 0.5)),
        motion_stability=float(candidate.get("motion_stability", 0.5)),
        seed=seed,
    )
    js = javis_score_proxy(
        av_sync=float(candidate.get("av_sync", 0.5)),
        fine_grained_match=float(candidate.get("fine_grained_match", 0.5)),
        seed=seed + 17,
    )
    return {
        "video_reward_ta": vr,
        "javis_score": js,
        "combined": combined_score(vr, js, vr_weight=vr_weight, javis_weight=javis_weight),
    }
