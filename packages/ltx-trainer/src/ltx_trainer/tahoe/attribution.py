"""Strategy attribution summaries for Logic Planning (Sec. 5.3–5.4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.tahoe.hints import Strategy, StrategyEvalStats


@dataclass(frozen=True)
class StrategyCredibility:
    strategy_id: str
    success_rate: float
    harm_rate: float
    inert_share: float
    support: int


def format_strategy_credibility(strategy: Strategy) -> StrategyCredibility:
    stats = strategy.eval_stats or StrategyEvalStats()
    support = len(stats.retrieved)
    if support == 0:
        return StrategyCredibility(strategy.strategy_id, 0.0, 0.0, 1.0, 0)
    helped = len(stats.helped)
    hurt = len(stats.hurt)
    inert = max(0, support - helped - hurt)
    return StrategyCredibility(
        strategy_id=strategy.strategy_id,
        success_rate=helped / support,
        harm_rate=hurt / support,
        inert_share=inert / support,
        support=support,
    )


def rank_strategies(strategies: list[Strategy]) -> list[Strategy]:
    """Prefer high success rate, then recency; down-rank high harm."""

    def _key(s: Strategy) -> tuple[float, float, float]:
        cred = format_strategy_credibility(s)
        return (cred.success_rate - cred.harm_rate, cred.success_rate, s.recency)

    return sorted(strategies, key=_key, reverse=True)
