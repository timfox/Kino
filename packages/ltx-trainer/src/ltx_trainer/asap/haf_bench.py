"""HAF-Bench metrics and taxonomy (Sec. 5, Eq. 7–8)."""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor


class HAFCategory(str, Enum):
    """Five anatomical prompt categories (100 prompts each in full bench)."""

    BASIC_ANATOMY = "basic_anatomy"
    SEMANTIC_GESTURES = "semantic_gestures"
    SELF_INTERACTION = "self_interaction"
    OBJECT_INTERACTION = "object_interaction"
    FULL_BODY = "full_body"


# Minimal curated examples for smoke tests / agent harness (full bench = 500 prompts).
HAF_BENCH_SAMPLE_PROMPTS: dict[HAFCategory, list[str]] = {
    HAFCategory.BASIC_ANATOMY: [
        "A young woman holding up one finger to test the wind direction.",
        "A little boy counting to ten, holding up both hands with all ten fingers spread.",
    ],
    HAFCategory.SEMANTIC_GESTURES: [
        "A nurse in scrubs making an OK gesture with slightly bent fingers in a hospital corridor.",
        "A cheerful girl in a school uniform holding up three fingers in front of a chalkboard.",
    ],
    HAFCategory.SELF_INTERACTION: [
        "A child resting her chin on her hand with fingers slightly curved, daydreaming in a classroom.",
        "A man standing on a balcony full body, leaning his elbows on the railing and clasping his hands.",
    ],
    HAFCategory.OBJECT_INTERACTION: [
        "A man lying on the beach giving a relaxed thumbs-up.",
        "A young girl sitting on a swing, gently curling her finger to beckon someone over.",
    ],
    HAFCategory.FULL_BODY: [
        "A man in athletic wear stretching both arms overhead in a sunny park.",
        "A dancer mid-leap with arms extended and legs in split position on stage.",
    ],
}


def anatomical_error_rate(
    fail_flags: Tensor,
) -> float:
    """Eq. (7): AER = fraction of samples where VLM judge fails."""
    if fail_flags.numel() == 0:
        return 0.0
    return float(fail_flags.float().mean().item())


def relative_superiority_index(
    n_win: int,
    n_tie: int,
    n_lose: int,
) -> float:
    """Eq. (8): RSI = (N_win + N_tie) / (N_lose + N_tie)."""
    denom = n_lose + n_tie
    if denom <= 0:
        return float("inf") if (n_win + n_tie) > 0 else 1.0
    return (n_win + n_tie) / denom


def vlm_judge_stub(
    images: Tensor,
    *,
    threshold: float = 0.5,
) -> Tensor:
    """Placeholder VLM anatomical pass/fail when Gemini judge is external.

    Uses mean luminance as deterministic pseudo-score for tests only.
    Returns bool tensor: True = Fail (anatomical error).
    """
    score = images.float().mean(dim=(-3, -2, -1))
    return score < threshold


def category_aer_report(
    category_fail: dict[HAFCategory, Tensor],
) -> dict[str, float]:
    """Per-category AER for radar plots (Fig. 4)."""
    return {cat.value: anatomical_error_rate(flags) for cat, flags in category_fail.items()}
