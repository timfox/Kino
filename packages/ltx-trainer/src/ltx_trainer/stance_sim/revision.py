"""Counterfactual context revision strategies (arXiv:2606.06443)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ltx_trainer.stance_sim.metrics import stance_score


class RevisionStrategy(str, Enum):
    PARAPHRASE = "paraphrase"
    EXPLAIN = "explain"
    ADD = "add"
    MEME = "meme"
    MEME_WHITE = "r_white_meme"
    MEME_HUMOR = "r_humor"
    MEME_CAPTION_CUT = "r_caption_cut"
    MEME_CAPTION = "r_caption"


# Toy shift priors on ordinal stance when revising last other-user message (CPU smoke).
_STRATEGY_SHIFT: dict[RevisionStrategy, float] = {
    RevisionStrategy.PARAPHRASE: 0.0,
    RevisionStrategy.EXPLAIN: 0.25,
    RevisionStrategy.ADD: 0.55,
    RevisionStrategy.MEME: 0.60,
    RevisionStrategy.MEME_WHITE: 0.48,
    RevisionStrategy.MEME_HUMOR: 0.35,
    RevisionStrategy.MEME_CAPTION_CUT: 0.40,
    RevisionStrategy.MEME_CAPTION: 0.38,
}

# Tone revision: add depolarizes toward midpoint; meme amplifies extremes.
_TONE_SHIFT: dict[RevisionStrategy, tuple[float, float]] = {
    RevisionStrategy.ADD: (0.35, -0.30),  # pull toward 50 from below/above
    RevisionStrategy.MEME: (0.45, 0.40),  # push away from midpoint
}


@dataclass(frozen=True)
class ConversationInstance:
    """Minimal Reddit thread slice for stance audit."""

    instance_id: str
    target_user: str
    stance_target: str
    context: str
    last_other_message: str
    last_target_message: str
    observed_stance: str
    subreddit: str = "MachineLearning"


def revise_message(
    message: str,
    strategy: RevisionStrategy | str,
    *,
    seed: int = 0,
) -> str:
    """Deterministic toy revision — prefixes strategy tag; real pipeline uses LLM prompts."""
    strat = RevisionStrategy(strategy) if not isinstance(strategy, RevisionStrategy) else strategy
    rng = np.random.default_rng(seed)
    tag = strat.value.replace("r_", "")
    jitter = rng.integers(0, 1000)
    return f"[{tag}:{jitter}] {message.strip()}"


def revise_tone(original_tone: float, strategy: RevisionStrategy | str) -> float:
    """Toy LIWC Tone update for add vs meme mechanism analysis."""
    strat = RevisionStrategy(strategy) if not isinstance(strategy, RevisionStrategy) else strategy
    if strat not in _TONE_SHIFT:
        return original_tone
    toward_mid, away_mid = _TONE_SHIFT[strat]
    midpoint = 50.0
    if original_tone <= midpoint:
        if strat == RevisionStrategy.ADD:
            return min(midpoint, original_tone + toward_mid * (midpoint - original_tone))
        return max(0.0, min(100.0, original_tone - away_mid * original_tone))
    if strat == RevisionStrategy.ADD:
        return max(midpoint, original_tone - toward_mid * (original_tone - midpoint))
    return max(0.0, min(100.0, original_tone + away_mid * (100.0 - original_tone)))


def simulate_revised_stance(
    inferred_stance: str,
    strategy: RevisionStrategy | str,
    *,
    seed: int = 0,
    noise: float = 0.05,
) -> str:
    """Map inferred stance through strategy prior + noise to revised label."""
    strat = RevisionStrategy(strategy) if not isinstance(strategy, RevisionStrategy) else strategy
    base = float(stance_score(inferred_stance))
    shift = _STRATEGY_SHIFT.get(strat, 0.0)
    rng = np.random.default_rng(seed)
    score = base + shift + rng.normal(0.0, noise)
    if score >= 0.35:
        return "positive"
    if score <= -0.35:
        return "negative"
    return "neutral"


def meme_text_payload(template_id: int, *, top: str, bottom: str = "") -> dict[str, Any]:
    """ImgFlip-style meme text slots (Appendix G.4)."""
    return {
        "template_id": template_id,
        "top_text": top,
        "bottom_text": bottom,
        "positions": {"top_text": top, "bottom_text": bottom} if bottom else {"top_text": top},
    }
