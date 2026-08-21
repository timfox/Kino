"""Hint-guided inference: Logic Planning + SQL Synthesis (Sec. 5.4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.tahoe.attribution import format_strategy_credibility
from ltx_trainer.tahoe.hints import HintBank, SyntaxHint
from ltx_trainer.tahoe.retrieval import retrieve_semantic_hints, selected_strategies


@dataclass(frozen=True)
class LogicPlan:
    question: str
    strategy_ids: tuple[str, ...]
    credibility: tuple[dict[str, float], ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class InferenceResult:
    logic_plan: LogicPlan
    sql: str
    syntax_hints_injected: int


def build_logic_plan(
    question: str,
    bank: HintBank,
    *,
    database_id: str | None = None,
    user_id: str | None = None,
) -> LogicPlan:
    retrieved = retrieve_semantic_hints(
        question, bank, database_id=database_id, user_id=user_id
    )
    strategies = selected_strategies(retrieved)
    creds = tuple(
        {
            "success_rate": c.success_rate,
            "harm_rate": c.harm_rate,
            "inert_share": c.inert_share,
            "support": float(c.support),
        }
        for c in (format_strategy_credibility(s) for s in strategies)
    )
    notes = tuple(s.preferred_action for s in strategies[:3])
    return LogicPlan(
        question=question,
        strategy_ids=tuple(s.strategy_id for s in strategies[:3]),
        credibility=creds,
        notes=notes,
    )


def synthesize_sql(
    question: str,
    plan: LogicPlan,
    syntax_hints: tuple[SyntaxHint, ...],
) -> str:
    """Deterministic stub synthesis for paper walkthroughs."""
    q = question.lower()
    quote_rule = any("quote" in h.rule.lower() for h in syntax_hints)
    if "count" in q and "order" in q and quote_rule:
        return 'SELECT COUNT(*) AS "total"\nFROM "SALES"."ORDERS";'
    if "log10" in q or "log 10" in q:
        return (
            'SELECT "event_type", LOG(10, "view_count" + 1) AS "log_views"\n'
            'FROM "ANALYTICS"."EVENTS";'
        )
    if "visitor" in q or "unique" in q:
        return (
            'SELECT DISTINCT "USER_PSEUDO_ID"\n'
            'FROM "GA4"."EVENTS"\n'
            "WHERE \"EVENT_DATE\" >= DATEADD(month, -1, CURRENT_DATE());"
        )
    if "top" in q and "product" in q and "tie" in q:
        return (
            'WITH "top_sales" AS (\n'
            '  SELECT MAX("sales") AS "max_sales" FROM "PRODUCTS"\n'
            ")\n"
            'SELECT "name", "sales" FROM "PRODUCTS" p\n'
            'JOIN "top_sales" t ON p."sales" = t."max_sales";'
        )
    return f"-- Tahoe stub: {plan.strategy_ids or ('no-strategy',)}"


def hint_guided_inference(
    question: str,
    bank: HintBank,
    *,
    database_id: str | None = None,
    user_id: str | None = None,
    dialect: str = "snowflake",
) -> InferenceResult:
    plan = build_logic_plan(question, bank, database_id=database_id, user_id=user_id)
    syntax = bank.dialect_syntax(dialect)
    sql = synthesize_sql(question, plan, syntax)
    return InferenceResult(
        logic_plan=plan,
        sql=sql,
        syntax_hints_injected=len(syntax),
    )
