#!/usr/bin/env python3
"""Evaluate SpaceDG-Bench predictions (JSONL)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.spacedg.eval import evaluate_predictions, score_qa  # noqa: E402
from ltx_trainer.spacedg.schema import AnswerFormat, QAPair, QuestionType, SpatialTaskGroup  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="SpaceDG-Bench eval")
    p.add_argument("predictions", help="JSONL with question, answer, prediction, degradation")
    args = p.parse_args()
    items: list[tuple] = []
    for line in Path(args.predictions).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qa = QAPair(
            question=row["question"],
            answer=row["answer"],
            question_type=QuestionType(row.get("question_type", "object_counting")),
            task_group=SpatialTaskGroup(row.get("task_group", "object_centric")),
            answer_format=AnswerFormat(row.get("answer_format", "numeric")),
        )
        items.append((qa, row.get("degradation", "original"), row["prediction"]))
    result = evaluate_predictions(items)
    acc = 100.0 * result.correct / max(result.total, 1)
    print(f"Overall: {acc:.2f}% ({result.correct:.0f}/{result.total})")
    for deg, scores in sorted(result.by_degradation.items()):
        print(f"  {deg}: {100.0 * sum(scores) / len(scores):.2f}%")


if __name__ == "__main__":
    main()
