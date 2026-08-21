"""Ranking accuracy metric for scheduling (Eq. 2, Sec. 4.1)."""

from __future__ import annotations

from ltx_trainer.clairvoyant.constants import MEDIUM_MAX_TOKENS, SHORT_MAX_TOKENS
from ltx_trainer.clairvoyant.predictor import predict_plong


def token_class(num_tokens: int) -> str:
    if num_tokens < SHORT_MAX_TOKENS:
        return "short"
    if num_tokens < MEDIUM_MAX_TOKENS:
        return "medium"
    return "long"


def pairwise_ranking_accuracy(
    short_plongs: list[float],
    long_plongs: list[float],
) -> float:
    """
    Fraction of (Short, Long) pairs where P(Long)_long > P(Long)_short.
    """
    if not short_plongs or not long_plongs:
        return 0.0
    correct = 0
    total = len(short_plongs) * len(long_plongs)
    for ps in short_plongs:
        for pl in long_plongs:
            if pl > ps:
                correct += 1
    return correct / total


def ranking_accuracy_from_labeled(
    prompts: list[str],
    response_token_lens: list[int],
    *,
    variant: str = "sharegpt",
) -> dict[str, float | int]:
    short_pl: list[float] = []
    long_pl: list[float] = []
    for prompt, ntok in zip(prompts, response_token_lens, strict=True):
        pl = predict_plong(prompt, variant=variant)
        cls = token_class(ntok)
        if cls == "short":
            short_pl.append(pl)
        elif cls == "long":
            long_pl.append(pl)
    acc = pairwise_ranking_accuracy(short_pl, long_pl)
    return {
        "ranking_accuracy": round(acc, 4),
        "n_short": len(short_pl),
        "n_long": len(long_pl),
        "n_pairs": len(short_pl) * len(long_pl),
    }
