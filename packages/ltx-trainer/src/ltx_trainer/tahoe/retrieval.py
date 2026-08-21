"""Scope-aware semantic hint retrieval (Sec. 5.4)."""

from __future__ import annotations

from ltx_trainer.tahoe.hints import HintBank, SemanticHint, Strategy
from ltx_trainer.tahoe.attribution import rank_strategies


def filter_by_scope(
    hints: tuple[SemanticHint, ...],
    *,
    database_id: str | None = None,
    user_id: str | None = None,
) -> list[SemanticHint]:
    out: list[SemanticHint] = []
    for hint in hints:
        if hint.scope == "general":
            out.append(hint)
        elif hint.scope == "database" and database_id and hint.database_id == database_id:
            out.append(hint)
        elif hint.scope == "user" and user_id and hint.user_id == user_id:
            out.append(hint)
    return out


def _trigger_match(question: str, hint: SemanticHint) -> bool:
    q = question.lower()
    needles = [hint.trigger.lower(), *(a.lower() for a in hint.aliases)]
    return any(n in q or q in n for n in needles if n)


def retrieve_semantic_hints(
    question: str,
    bank: HintBank,
    *,
    database_id: str | None = None,
    user_id: str | None = None,
) -> list[SemanticHint]:
    """Stub retriever: lexical trigger overlap (paper uses LLM retrieval)."""
    scoped = filter_by_scope(bank.semantic, database_id=database_id, user_id=user_id)
    return [h for h in scoped if _trigger_match(question, h)]


def selected_strategies(retrieved: list[SemanticHint]) -> list[Strategy]:
    strategies: list[Strategy] = []
    for hint in retrieved:
        strategies.extend(hint.strategies)
    return rank_strategies(strategies)
