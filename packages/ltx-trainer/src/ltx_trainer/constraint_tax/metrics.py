"""Constraint tax and deployment metrics (Sec. 4)."""

from __future__ import annotations

from dataclasses import dataclass


def constraint_tax(acc_baseline: float, acc_constrained: float) -> float:
    """Eq. (2): clipped accuracy loss."""
    return max(0.0, float(acc_baseline) - float(acc_constrained))


def constraint_tax_normalized(
    acc_baseline: float,
    acc_constrained: float,
    *,
    epsilon: float = 1e-9,
) -> float:
    """Eq. (3): normalized tax relative to baseline accuracy."""
    base = max(epsilon, float(acc_baseline))
    return constraint_tax(acc_baseline, acc_constrained) / base


def wrong_valid_schema_rate(
    *,
    schema_valid: bool,
    semantically_correct: bool,
) -> bool:
    """True when output passes schema but fails answer/executable check."""
    return bool(schema_valid and not semantically_correct)


@dataclass
class AggregateRates:
    answer_accuracy: float
    schema_validity: float
    executable_accuracy: float
    wrong_valid_schema_pct: float

    def to_dict(self) -> dict[str, float]:
        return {
            "answer_accuracy": self.answer_accuracy,
            "schema_validity": self.schema_validity,
            "executable_accuracy": self.executable_accuracy,
            "wrong_valid_schema_pct": self.wrong_valid_schema_pct,
        }


def aggregate_from_records(records: list[dict]) -> AggregateRates:
    """Compute Table-3-style rates from per-generation records."""
    n = max(1, len(records))
    ans = sum(1 for r in records if r.get("answer_correct")) / n
    val = sum(1 for r in records if r.get("schema_valid")) / n
    exe = sum(1 for r in records if r.get("executable_ok")) / n
    wvs = sum(1 for r in records if r.get("wrong_valid_schema")) / n
    return AggregateRates(
        answer_accuracy=round(100.0 * ans, 1),
        schema_validity=round(100.0 * val, 1),
        executable_accuracy=round(100.0 * exe, 1),
        wrong_valid_schema_pct=round(100.0 * wvs, 1),
    )
