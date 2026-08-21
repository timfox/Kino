"""Hint Bank data model (Sec. 5.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Scope = Literal["general", "database", "user"]


@dataclass(frozen=True)
class SyntaxHint:
    """Hsyn entry: (Rule, Example)."""

    hint_id: str
    rule: str
    example_schema: str
    example_sql: str


@dataclass
class StrategyEvalStats:
    """Post-learning attribution lists (Sec. 5.3)."""

    retrieved: list[str] = field(default_factory=list)
    helped: list[str] = field(default_factory=list)
    hurt: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Strategy:
    """Competing strategy under a semantic trigger."""

    strategy_id: str
    rationale: str
    preferred_action: str
    wrong_action: str
    recency: float
    eval_stats: StrategyEvalStats | None = None


@dataclass(frozen=True)
class SemanticHint:
    """Hsem entry: (Trigger, Scope, Strategies)."""

    hint_id: str
    trigger: str
    scope: Scope
    strategies: tuple[Strategy, ...]
    database_id: str | None = None
    user_id: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class HintBank:
    syntax: tuple[SyntaxHint, ...]
    semantic: tuple[SemanticHint, ...]

    def dialect_syntax(self, dialect: str = "snowflake") -> tuple[SyntaxHint, ...]:
        if dialect.lower() != "snowflake":
            return self.syntax
        return self.syntax
