"""Hint learning loop stub (Sec. 5.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DiffCategory = Literal[
    "JOIN_TYPE",
    "FILTER_SCOPE",
    "AGG_TIMING",
    "WINDOW_FRAME",
    "LIMIT_RANK",
    "FORMULA",
    "COLUMN_CHOICE",
    "UNIT_CAST",
    "SEMANTIC",
    "GLOBAL_PLAN",
    "OTHER",
]


@dataclass(frozen=True)
class AtomicDiff:
    diff_id: str
    phrase: str
    category: DiffCategory
    gold_strategy: str
    wrong_strategy: str
    impact: str


@dataclass
class TemporaryHintBank:
    """Per-example working copy ˜H during learning."""

    syntax_additions: list[str]
    semantic_additions: list[str]
    revisions: list[str]


def diff_from_syntax_error(error: str, wrong_sql: str, fixed_sql: str) -> AtomicDiff:
    return AtomicDiff(
        diff_id="GLOBAL::SYNTAX::1",
        phrase="GLOBAL",
        category="OTHER",
        gold_strategy="Apply dialect quoting rules from compiler feedback",
        wrong_strategy=f"Unquoted identifiers: {wrong_sql[:80]}",
        impact=error[:120],
    )


def diff_from_semantic_mismatch(phrase: str, gold: str, wrong: str) -> AtomicDiff:
    return AtomicDiff(
        diff_id=f"{phrase}::FORMULA::1",
        phrase=phrase,
        category="FORMULA",
        gold_strategy=gold,
        wrong_strategy=wrong,
        impact="Execution result mismatch vs ground truth",
    )


def should_merge_deltas(
    *,
    all_samples_correct: bool,
    correct_under_temp: int,
    correct_under_frozen: int,
    max_iterations_reached: bool,
) -> bool:
    """Stop criteria from Sec. 5.2.3."""
    if all_samples_correct:
        return True
    if max_iterations_reached:
        return correct_under_temp > correct_under_frozen
    return False
