"""Baseline comparisons (SQLGenie-style RAG, Sec. 6.4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.tahoe.benchmarks import TABLE_2_MAIN, TABLE_3_HELD_OUT


@dataclass(frozen=True)
class ExampleBankEntry:
    question: str
    sql: str
    masked_question: str


def masked_question(question: str) -> str:
    """Stub mask: lowercase + collapse whitespace."""
    return " ".join(question.lower().split())


def example_bank_from_development() -> tuple[ExampleBankEntry, ...]:
    return (
        ExampleBankEntry(
            question="Count all orders placed",
            sql='SELECT COUNT(*) AS "total"\nFROM "SALES"."ORDERS";',
            masked_question=masked_question("Count all orders placed"),
        ),
        ExampleBankEntry(
            question="Compute the log10-transformed view count per event type",
            sql='SELECT "event_type", LOG(10, "view_count" + 1) AS "log_views" FROM "ANALYTICS"."EVENTS";',
            masked_question=masked_question("Compute the log10-transformed view count per event type"),
        ),
    )


def retrieve_top_k(question: str, bank: tuple[ExampleBankEntry, ...], k: int = 3) -> list[ExampleBankEntry]:
    q = masked_question(question)
    scored = sorted(
        bank,
        key=lambda e: _token_overlap(q, e.masked_question),
        reverse=True,
    )
    return scored[:k]


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def cross_model_transfer_summary() -> list[dict[str, object]]:
    """Table 2 pass-rate deltas for shared Hint Bank."""
    out: list[dict[str, float]] = []
    for backbone in ("Doubao-2.0-lite", "GPT-5", "GPT-5.5"):
        vanilla = next(r for r in TABLE_2_MAIN if r["backbone"] == backbone and r["config"] == "Vanilla")
        tahoe = next(r for r in TABLE_2_MAIN if r["backbone"] == backbone and r["config"] == "Tahoe")
        out.append(
            {
                "backbone": backbone,
                "pass_rate_delta_pp": round((tahoe["pass_rate"] - vanilla["pass_rate"]) * 100, 2),
                "pass_at_4_delta_pp": round((tahoe["pass_at_4"] - vanilla["pass_at_4"]) * 100, 2),
            }
        )
    return out


def held_out_baseline_comparison() -> dict[str, object]:
    vanilla = TABLE_3_HELD_OUT[0]
    rag = TABLE_3_HELD_OUT[1]
    tahoe = TABLE_3_HELD_OUT[2]
    return {
        "vanilla": vanilla,
        "sqlgenie_style_rag": rag,
        "tahoe": tahoe,
        "rag_vs_vanilla_pass_rate_pp": round((rag["pass_rate"] - vanilla["pass_rate"]) * 100, 2),
        "tahoe_vs_vanilla_pass_rate_pp": round((tahoe["pass_rate"] - vanilla["pass_rate"]) * 100, 2),
        "tahoe_syntax_gain_pp": round((tahoe["syntax_pass"] - vanilla["syntax_pass"]) * 100, 2),
    }
