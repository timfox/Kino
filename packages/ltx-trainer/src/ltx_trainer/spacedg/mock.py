"""SpaceDG-Bench schema + scoring smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke() -> dict[str, Any]:
    schema = load_sibling(__file__, "schema")
    qa = schema.QAPair(
        question="How many objects?",
        answer="3",
        answer_format=schema.AnswerFormat.NUMERIC,
        question_type=schema.QuestionType.OBJECT_COUNTING,
        task_group=schema.SpatialTaskGroup.OBJECT_CENTRIC,
    )
    pred, gt = 3.0, 3.0
    rel = abs(pred - gt) / abs(gt)
    thetas = [0.5 + 0.05 * i for i in range(10)]
    mra = sum(1.0 for t in thetas if rel < (1.0 - t)) / len(thetas)
    return {
        "paper": "arXiv:2605.26061",
        "degradation": schema.DegradationType.LOW_LIGHT.value,
        "mean_relative_accuracy": round(mra, 4),
        "exact_match": 1.0 if str(int(pred)) == qa.answer else 0.0,
    }
