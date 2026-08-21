"""Evaluation metrics: accuracy + perception alignment (Sec. 3.4)."""

from __future__ import annotations

from typing import Sequence

EntailmentOutcome = str  # entail | likely entail | not entail


def accuracy(predictions: Sequence[str], gold: Sequence[str]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, gold, strict=True) if p == g)
    return 100.0 * correct / len(gold)


def entailment_score(outcome: EntailmentOutcome) -> float:
    mapping = {"entail": 1.0, "likely entail": 0.5, "not entail": 0.0}
    return mapping.get(outcome.lower(), 0.0)


def perception_alignment(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def delta_accuracy(before: float, after: float) -> float:
    return after - before
