"""Strategy Attribution pass (Sec. 5.3 step 3)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ltx_trainer.tahoe.hints import HintBank, SemanticHint, Strategy, StrategyEvalStats


class AttributionVerdict(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    INERT = "inert"


@dataclass(frozen=True)
class AttributionExample:
    example_id: str
    question: str
    predicted_sql: str
    gold_sql: str
    execution_match: bool
    retrieved_hint_ids: tuple[str, ...]


def judge_strategy_application(
    example: AttributionExample,
    strategy: Strategy,
) -> AttributionVerdict:
    """SQL-evidence stub aligned with Sec. 5.3 verdict rules."""
    pred = example.predicted_sql.upper()
    gold = example.gold_sql.upper()
    preferred = strategy.preferred_action.upper()
    wrong = strategy.wrong_action.upper()

    if "USER_PSEUDO_ID" in preferred and "USER_PSEUDO_ID" in pred:
        return AttributionVerdict.POSITIVE if example.execution_match else AttributionVerdict.NEGATIVE
    if "USER_ID" in wrong and "USER_ID" in pred and "USER_PSEUDO_ID" not in pred:
        return AttributionVerdict.NEGATIVE
    if "LOG(10" in preferred and "+ 1" in pred:
        return AttributionVerdict.POSITIVE if example.execution_match else AttributionVerdict.INERT
    if "NULLIF" in pred and "+ 1" in gold:
        return AttributionVerdict.NEGATIVE
    if "LIMIT 1" in pred and "MAX" in gold:
        return AttributionVerdict.NEGATIVE
    if "MAX" in pred and example.execution_match:
        return AttributionVerdict.POSITIVE
    if strategy.strategy_id in {s.strategy_id for h in _hints_for_ids(example.retrieved_hint_ids) for s in h.strategies}:
        return AttributionVerdict.INERT
    return AttributionVerdict.INERT


def _hints_for_ids(hint_ids: tuple[str, ...]) -> list[SemanticHint]:
    from ltx_trainer.tahoe.running_example import seed_hint_bank

    bank = seed_hint_bank()
    by_id = {h.hint_id: h for h in bank.semantic}
    return [by_id[i] for i in hint_ids if i in by_id]


def _update_stats(stats: StrategyEvalStats | None, example_id: str, verdict: AttributionVerdict) -> StrategyEvalStats:
    base = stats or StrategyEvalStats()
    retrieved = list(base.retrieved)
    helped = list(base.helped)
    hurt = list(base.hurt)
    if example_id not in retrieved:
        retrieved.append(example_id)
    if verdict == AttributionVerdict.POSITIVE and example_id not in helped:
        helped.append(example_id)
    if verdict == AttributionVerdict.NEGATIVE and example_id not in hurt:
        hurt.append(example_id)
    return StrategyEvalStats(retrieved=retrieved, helped=helped, hurt=hurt)


def apply_attribution_to_hint(hint: SemanticHint, examples: list[AttributionExample]) -> SemanticHint:
    strategies: list[Strategy] = []
    for strategy in hint.strategies:
        stats = strategy.eval_stats
        for ex in examples:
            if hint.hint_id not in ex.retrieved_hint_ids:
                continue
            verdict = judge_strategy_application(ex, strategy)
            stats = _update_stats(stats, ex.example_id, verdict)
        strategies.append(replace(strategy, eval_stats=stats))
    return replace(hint, strategies=tuple(strategies))


def run_strategy_attribution(
    bank: HintBank,
    examples: list[AttributionExample] | None = None,
) -> HintBank:
    if examples is None:
        examples = [
            AttributionExample(
                example_id="ex_log10_1",
                question="log10 view count",
                predicted_sql='SELECT LOG(10, "view_count" + 1) FROM "T";',
                gold_sql='SELECT LOG(10, "view_count" + 1) FROM "T";',
                execution_match=True,
                retrieved_hint_ids=("sem_log10_counts",),
            ),
            AttributionExample(
                example_id="ex_ga4_1",
                question="unique visitors",
                predicted_sql='SELECT "USER_PSEUDO_ID" FROM "GA4"."EVENTS";',
                gold_sql='SELECT "USER_PSEUDO_ID" FROM "GA4"."EVENTS";',
                execution_match=True,
                retrieved_hint_ids=("sem_ga4_visitor_id",),
            ),
            AttributionExample(
                example_id="ex_ties_4",
                question="top product",
                predicted_sql="SELECT name FROM PRODUCTS ORDER BY sales DESC LIMIT 1",
                gold_sql='SELECT "name" FROM "PRODUCTS" WHERE "sales" = (SELECT MAX("sales") FROM "PRODUCTS")',
                execution_match=False,
                retrieved_hint_ids=("sem_top_product_ties",),
            ),
        ]
    semantic = tuple(apply_attribution_to_hint(h, examples) for h in bank.semantic)
    return HintBank(syntax=bank.syntax, semantic=semantic)


def attribution_summary(bank: HintBank) -> list[dict[str, object]]:
    from ltx_trainer.tahoe.attribution import format_strategy_credibility

    rows: list[dict[str, object]] = []
    for hint in bank.semantic:
        for strategy in hint.strategies:
            cred = format_strategy_credibility(strategy)
            rows.append(
                {
                    "hint_id": hint.hint_id,
                    "strategy_id": strategy.strategy_id,
                    **cred.__dict__,
                }
            )
    return rows
