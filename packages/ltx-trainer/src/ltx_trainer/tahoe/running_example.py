"""Sec. 5.6 case-study hints + seed Hint Bank."""

from __future__ import annotations

from ltx_trainer.tahoe.hints import HintBank, SemanticHint, Strategy, StrategyEvalStats, SyntaxHint
from ltx_trainer.tahoe.inference import hint_guided_inference


def paper_syntax_hints() -> tuple[SyntaxHint, ...]:
    return (
        SyntaxHint(
            hint_id="syn_quote_identifiers",
            rule=(
                "Quote every database, schema, table, column, CTE, and alias exactly as stored; "
                "quote each element of a fully-qualified path separately."
            ),
            example_schema='sales.orders("orderId", order_date)',
            example_sql='SELECT COUNT(*) AS "total"\nFROM "SALES"."ORDERS";',
        ),
        SyntaxHint(
            hint_id="syn_snowflake_uppercase",
            rule="Unquoted identifiers are uppercased in Snowflake; use double quotes for mixed case.",
            example_schema="PRODUCTS(name, sales)",
            example_sql='SELECT "name", "sales" FROM "PRODUCTS";',
        ),
    )


def paper_semantic_hints() -> tuple[SemanticHint, ...]:
    log10_stats = StrategyEvalStats(
        retrieved=["ex_log10_1", "ex_log10_2", "ex_log10_3"],
        helped=["ex_log10_1", "ex_log10_2"],
        hurt=[],
    )
    ga4_stats = StrategyEvalStats(
        retrieved=["ex_ga4_1", "ex_ga4_2"],
        helped=["ex_ga4_1"],
        hurt=[],
    )
    ties_stats = StrategyEvalStats(
        retrieved=["ex_ties_1", "ex_ties_2", "ex_ties_3", "ex_ties_4"],
        helped=["ex_ties_1", "ex_ties_2", "ex_ties_3"],
        hurt=["ex_ties_4"],
    )
    return (
        SemanticHint(
            hint_id="sem_log10_counts",
            trigger="log10 transformation of counts",
            scope="general",
            aliases=("log10", "log 10", "logarithm"),
            strategies=(
                Strategy(
                    strategy_id="log10_add_one",
                    rationale=(
                        "When log10 is applied to count data that may contain zeros, "
                        "add 1 before the log to avoid -infinity."
                    ),
                    preferred_action='LOG(10, "{COLUMN}" + 1)',
                    wrong_action='LOG(10, NULLIF("{COLUMN}", 0))',
                    recency=3.0,
                    eval_stats=log10_stats,
                ),
            ),
        ),
        SemanticHint(
            hint_id="sem_ga4_visitor_id",
            trigger="reference to column USER_ID in GA4 events tables",
            scope="database",
            database_id="GA4",
            aliases=("unique visitors", "visitor id", "user_id"),
            strategies=(
                Strategy(
                    strategy_id="ga4_pseudo_id",
                    rationale=(
                        "GA4 sample ecommerce tables store visitor IDs in USER_PSEUDO_ID; "
                        "USER_ID is almost always NULL."
                    ),
                    preferred_action='Filter on USER_PSEUDO_ID instead of USER_ID',
                    wrong_action='WHERE USER_ID = ...',
                    recency=2.0,
                    eval_stats=ga4_stats,
                ),
            ),
        ),
        SemanticHint(
            hint_id="sem_top_product_ties",
            trigger="top-selling product",
            scope="general",
            aliases=("top product", "best seller", "highest sales"),
            strategies=(
                Strategy(
                    strategy_id="top_with_ties",
                    rationale="Return all rows tied at the maximum sales value, not LIMIT 1.",
                    preferred_action="Filter with MAX(sales) subquery or DENSE_RANK",
                    wrong_action="ORDER BY sales DESC LIMIT 1",
                    recency=4.0,
                    eval_stats=ties_stats,
                ),
            ),
        ),
    )


def seed_hint_bank() -> HintBank:
    """Development-phase bank (11 syntax + 37 semantic in paper; stub uses core set)."""
    syntax = paper_syntax_hints()
    # Pad metadata to paper counts via placeholder ids (not full hint text in stub)
    extra_syntax = tuple(
        SyntaxHint(
            hint_id=f"syn_stub_{i}",
            rule=f"Snowflake dialect rule stub {i}",
            example_schema="",
            example_sql="SELECT 1;",
        )
        for i in range(3, 12)
    )
    semantic = paper_semantic_hints()
    extra_semantic = tuple(
        SemanticHint(
            hint_id=f"sem_stub_{i}",
            trigger=f"stub trigger {i}",
            scope="general" if i % 2 else "database",
            database_id=None if i % 2 else "SF_LOCAL",
            strategies=(
                Strategy(
                    strategy_id=f"stub_strategy_{i}",
                    rationale="Placeholder semantic hint from development batch",
                    preferred_action="stub",
                    wrong_action="stub",
                    recency=float(i),
                ),
            ),
        )
        for i in range(4, 38)
    )
    return HintBank(syntax=syntax + extra_syntax, semantic=semantic + extra_semantic)


def demo_snowflake_quoting() -> dict[str, object]:
    bank = seed_hint_bank()
    result = hint_guided_inference("Count all orders placed", bank)
    return {
        "sql": result.sql,
        "quoted_identifiers": '"SALES"' in result.sql and '"ORDERS"' in result.sql,
        "syntax_hints": result.syntax_hints_injected,
    }


def demo_log10_transformation() -> dict[str, object]:
    bank = seed_hint_bank()
    result = hint_guided_inference(
        "Compute the log10-transformed view count per event type", bank
    )
    return {
        "sql": result.sql,
        "uses_add_one": "+ 1" in result.sql,
        "strategy_ids": result.logic_plan.strategy_ids,
    }


def demo_ga4_visitors() -> dict[str, object]:
    bank = seed_hint_bank()
    result = hint_guided_inference(
        "List unique visitors from last month",
        bank,
        database_id="GA4",
    )
    return {
        "sql": result.sql,
        "uses_pseudo_id": "USER_PSEUDO_ID" in result.sql,
        "strategy_ids": result.logic_plan.strategy_ids,
    }


def demo_top_product_ties() -> dict[str, object]:
    bank = seed_hint_bank()
    result = hint_guided_inference(
        "Show me the top-selling product including ties",
        bank,
    )
    return {
        "sql": result.sql,
        "avoids_limit_one": "LIMIT 1" not in result.sql.upper(),
        "uses_max_join": "MAX" in result.sql.upper(),
    }
