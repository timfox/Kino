"""Prediction scoring (SpaceDG-Bench protocol)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.spacedg.metrics import accuracy, list_numeric_score, mean_relative_accuracy
from ltx_trainer.spacedg.schema import AnswerFormat, QAPair


@dataclass
class EvalResult:
    correct: float
    total: int
    by_degradation: dict[str, list[float]]
    by_task_group: dict[str, list[float]]


def score_qa(prediction: str, qa: QAPair) -> float:
    if qa.answer_format == AnswerFormat.NUMERIC:
        try:
            return mean_relative_accuracy(float(prediction), float(qa.answer))
        except ValueError:
            return 0.0
    if qa.answer_format == AnswerFormat.LIST_NUMERIC:
        return list_numeric_score(prediction, qa.answer)
    return accuracy(prediction, qa.answer)


def evaluate_predictions(
    items: list[tuple[QAPair, str, str]],
) -> EvalResult:
    scores: list[float] = []
    by_deg: dict[str, list[float]] = {}
    by_grp: dict[str, list[float]] = {}
    for qa, deg, pred in items:
        s = score_qa(pred, qa)
        scores.append(s)
        by_deg.setdefault(deg, []).append(s)
        by_grp.setdefault(qa.task_group.value, []).append(s)
    return EvalResult(
        correct=sum(scores),
        total=len(scores),
        by_degradation=by_deg,
        by_task_group=by_grp,
    )
