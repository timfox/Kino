"""Evaluate model outputs on LiveBrowseComp items."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ltx_trainer.livebrowsecomp.corpus import LiveBrowseCompItem
from ltx_trainer.livebrowsecomp.io import load_items
from ltx_trainer.livebrowsecomp.judge import grade_response, simulate_judge_output
from ltx_trainer.livebrowsecomp.metrics import avg_at_k, pass_at_k_from_bools


@dataclass
class QuestionEval:
    idx: int
    correct_samples: list[bool]
    pass_at_4: float
    avg_at_4: float
    predictions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "pass_at_4": round(self.pass_at_4, 4),
            "avg_at_4": round(self.avg_at_4, 4),
            "num_samples": len(self.correct_samples),
        }


@dataclass
class EvalReport:
    num_questions: int
    pass_at_4: float
    avg_at_4: float
    per_question: list[QuestionEval]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_questions": self.num_questions,
            "pass_at_4": round(self.pass_at_4, 4),
            "avg_at_4": round(self.avg_at_4, 4),
            "per_question": [q.to_dict() for q in self.per_question],
        }


def evaluate_item(
    item: LiveBrowseCompItem,
    samples: list[str],
    *,
    k: int = 4,
    use_simulated_judge: bool = True,
) -> QuestionEval:
    flags: list[bool] = []
    for resp in samples:
        judge_out = (
            simulate_judge_output(item.problem, resp, item.answer) if use_simulated_judge else None
        )
        jr = grade_response(
            item.problem,
            resp,
            item.answer,
            judge_output=judge_out,
        )
        flags.append(jr.correct)
    return QuestionEval(
        idx=item.idx,
        correct_samples=flags,
        pass_at_4=pass_at_k_from_bools(flags, k),
        avg_at_4=avg_at_k(flags, k),
        predictions=samples,
    )


def evaluate_predictions(
    items: list[LiveBrowseCompItem],
    predictions: dict[int, list[str]],
    *,
    k: int = 4,
) -> EvalReport:
    per_q: list[QuestionEval] = []
    for item in items:
        samples = predictions.get(item.idx, [])
        per_q.append(evaluate_item(item, samples, k=k))
    if not per_q:
        return EvalReport(0, 0.0, 0.0, [])
    pass_scores = [q.pass_at_4 for q in per_q]
    avg_scores = [q.avg_at_4 for q in per_q]
    return EvalReport(
        num_questions=len(per_q),
        pass_at_4=100.0 * sum(pass_scores) / len(pass_scores),
        avg_at_4=100.0 * sum(avg_scores) / len(avg_scores),
        per_question=per_q,
    )


def load_submission_jsonl(path: str | Path) -> dict[int, list[str]]:
    """Submission format: {idx, samples: [response, ...]} per line."""
    out: dict[int, list[str]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["idx"])
            samples = row.get("samples") or [row.get("response", "")]
            out[idx] = [str(s) for s in samples]
    return out


def evaluate_submission_file(
    submission_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    limit: int | None = None,
    k: int = 4,
    download: bool = True,
) -> EvalReport:
    items = load_items(dataset_path, limit=limit, download=download)
    preds = load_submission_jsonl(submission_path)
    return evaluate_predictions(items, preds, k=k)


def iter_builtin_smoke_predictions(items: list[LiveBrowseCompItem]) -> dict[int, list[str]]:
    """Four samples: one correct tag, three wrong."""
    out: dict[int, list[str]] = {}
    for item in items:
        good = f"Research complete. <answer>{item.answer}</answer>"
        bad = "<answer>unknown wrong answer</answer>"
        out[item.idx] = [good, bad, bad, bad]
    return out
