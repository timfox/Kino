"""Appendix A atomic diff schema + serialization."""

from __future__ import annotations

from ltx_trainer.tahoe.learning import AtomicDiff, DiffCategory


def atomic_diff_schema_fields() -> tuple[str, ...]:
    return (
        "DiffID",
        "Phrase",
        "Category",
        "StepRef",
        "GoldStrategy",
        "GoldSQL",
        "WrongStrategy",
        "WrongSQL",
        "Impact",
    )


def atomic_diff_to_record(diff: AtomicDiff, *, step_ref: str = "GLOBAL") -> dict[str, str]:
    """JSON-like record enforced during hint learning (Appendix A)."""
    return {
        "DiffID": diff.diff_id,
        "Phrase": diff.phrase,
        "Category": diff.category,
        "StepRef": step_ref,
        "GoldStrategy": diff.gold_strategy,
        "GoldSQL": "",
        "WrongStrategy": diff.wrong_strategy,
        "WrongSQL": "",
        "Impact": diff.impact,
    }


def make_limit_rank_diff() -> AtomicDiff:
    """Fig. 1 Case B: LIMIT 1 vs tie-handling."""
    return AtomicDiff(
        diff_id="top-selling product::LIMIT_RANK::1",
        phrase="top-selling product",
        category="LIMIT_RANK",
        gold_strategy="Return all products tied at maximum sales",
        wrong_strategy="ORDER BY sales DESC LIMIT 1 drops ties",
        impact="User feedback: incomplete result when multiple products share top sales",
    )


def make_column_choice_diff() -> AtomicDiff:
    return AtomicDiff(
        diff_id="unique visitors::COLUMN_CHOICE::1",
        phrase="unique visitors",
        category="COLUMN_CHOICE",
        gold_strategy="Use USER_PSEUDO_ID for GA4 visitor identity",
        wrong_strategy="Filter on USER_ID which is NULL in GA4 sample schema",
        impact="Execution returns empty set vs ground truth",
    )
