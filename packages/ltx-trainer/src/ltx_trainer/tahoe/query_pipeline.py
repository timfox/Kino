"""End-to-end hint-guided query with syntax critic (inference + compiler)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.tahoe.compiler import syntax_critic_loop
from ltx_trainer.tahoe.hints import HintBank
from ltx_trainer.tahoe.inference import hint_guided_inference


@dataclass(frozen=True)
class QueryPipelineResult:
    question: str
    raw_sql: str
    final_sql: str
    critic_rounds: int
    syntax_valid: bool
    strategy_ids: tuple[str, ...]
    syntax_hints_injected: int


def run_hint_guided_query(
    question: str,
    bank: HintBank,
    *,
    database_id: str | None = None,
    user_id: str | None = None,
    max_critic_rounds: int = 3,
) -> QueryPipelineResult:
    inferred = hint_guided_inference(
        question,
        bank,
        database_id=database_id,
        user_id=user_id,
    )
    final_sql, rounds, valid = syntax_critic_loop(
        inferred.sql,
        bank,
        max_rounds=max_critic_rounds,
    )
    return QueryPipelineResult(
        question=question,
        raw_sql=inferred.sql,
        final_sql=final_sql,
        critic_rounds=rounds,
        syntax_valid=valid,
        strategy_ids=inferred.logic_plan.strategy_ids,
        syntax_hints_injected=inferred.syntax_hints_injected,
    )
