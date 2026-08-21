"""Spider 2.0–Snow evaluation metrics (Sec. 6.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateOutcome:
    example_id: str
    sql: str
    execution_match: bool
    syntax_valid: bool
    critic_rounds: int


@dataclass(frozen=True)
class EvaluationMetrics:
    pass_rate: float
    pass_at_4: float
    syntax_pass_rate: float
    avg_critics: float
    examples: int
    candidates: int


def _group_by_example(outcomes: list[CandidateOutcome]) -> dict[str, list[CandidateOutcome]]:
    groups: dict[str, list[CandidateOutcome]] = {}
    for row in outcomes:
        groups.setdefault(row.example_id, []).append(row)
    return groups


def compute_metrics(outcomes: list[CandidateOutcome], *, k: int = 4) -> EvaluationMetrics:
    if not outcomes:
        return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0, 0)
    n = len(outcomes)
    pass_rate = sum(1 for o in outcomes if o.execution_match) / n
    syntax_pass = sum(1 for o in outcomes if o.syntax_valid) / n
    avg_critics = sum(o.critic_rounds for o in outcomes) / n
    groups = _group_by_example(outcomes)
    solved = 0
    for rows in groups.values():
        sample = rows[:k]
        if any(r.execution_match for r in sample):
            solved += 1
    pass_at_k = solved / len(groups) if groups else 0.0
    return EvaluationMetrics(
        pass_rate=pass_rate,
        pass_at_4=pass_at_k,
        syntax_pass_rate=syntax_pass,
        avg_critics=avg_critics,
        examples=len(groups),
        candidates=n,
    )


def metrics_from_table_row(row: dict[str, float]) -> EvaluationMetrics:
    return EvaluationMetrics(
        pass_rate=float(row["pass_rate"]),
        pass_at_4=float(row["pass_at_4"]),
        syntax_pass_rate=float(row["syntax_pass"]),
        avg_critics=float(row["avg_critics"]),
        examples=113,
        candidates=113 * 4,
    )
