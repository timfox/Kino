"""Hint Bank management: merge, dedup, vanilla coverage (Sec. 5.3)."""

from __future__ import annotations

from ltx_trainer.tahoe.hints import HintBank, SemanticHint, Strategy, SyntaxHint


def merge_syntax(existing: tuple[SyntaxHint, ...], incoming: tuple[SyntaxHint, ...]) -> tuple[SyntaxHint, ...]:
    by_id = {h.hint_id: h for h in existing}
    for hint in incoming:
        by_id[hint.hint_id] = hint
    return tuple(by_id.values())


def merge_triggers(existing: SemanticHint, incoming: SemanticHint) -> SemanticHint:
    """Append non-conflicting strategies under the same trigger."""
    by_id = {s.strategy_id: s for s in existing.strategies}
    for strategy in incoming.strategies:
        if strategy.strategy_id not in by_id:
            by_id[strategy.strategy_id] = strategy
    return SemanticHint(
        hint_id=existing.hint_id,
        trigger=existing.trigger,
        scope=existing.scope,
        strategies=tuple(by_id.values()),
        database_id=existing.database_id,
        user_id=existing.user_id,
        aliases=existing.aliases,
    )


def batch_merge(bank: HintBank, deltas: list[SemanticHint]) -> HintBank:
    by_trigger: dict[tuple[str, str, str | None, str | None], SemanticHint] = {}
    for hint in bank.semantic:
        key = (hint.trigger, hint.scope, hint.database_id, hint.user_id)
        by_trigger[key] = hint
    for delta in deltas:
        key = (delta.trigger, delta.scope, delta.database_id, delta.user_id)
        if key in by_trigger:
            by_trigger[key] = merge_triggers(by_trigger[key], delta)
        else:
            by_trigger[key] = delta
    return HintBank(syntax=bank.syntax, semantic=tuple(by_trigger.values()))


def batch_merge_hints(
    bank: HintBank,
    *,
    syntax_deltas: tuple[SyntaxHint, ...] = (),
    semantic_deltas: tuple[SemanticHint, ...] = (),
) -> HintBank:
    merged_semantic = batch_merge(bank, list(semantic_deltas)).semantic
    merged_syntax = merge_syntax(bank.syntax, syntax_deltas)
    return HintBank(syntax=merged_syntax, semantic=merged_semantic)


def vanilla_strategy_from_success(strategy_id: str, rationale: str) -> Strategy:
    """Coverage assurance: distill first-pass success into competing strategy."""
    return Strategy(
        strategy_id=strategy_id,
        rationale=rationale,
        preferred_action="Model succeeded without explicit correction hint",
        wrong_action="N/A",
        recency=0.0,
        eval_stats=None,
    )
