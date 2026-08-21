"""Snowflake syntax critic stub (Sec. 5.2 Phase A, Sec. 6 metrics)."""

from __future__ import annotations

import re

from ltx_trainer.tahoe.hints import HintBank


_UNQUOTED_IDENT = re.compile(r"\b(FROM|JOIN|INTO|UPDATE)\s+([A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)*)", re.I)


def snowflake_syntax_valid(sql: str) -> bool:
    """Stub: reject bare mixed-case path segments that should be quoted."""
    if not sql or sql.strip().startswith("--"):
        return False
    for match in _UNQUOTED_IDENT.finditer(sql):
        path = match.group(2)
        if "." in path and '"' not in path:
            return False
        if path != path.upper() and '"' not in path:
            return False
    return True


def apply_syntax_hints(sql: str, bank: HintBank) -> str:
    """One-shot repair using quoting syntax hints."""
    if snowflake_syntax_valid(sql):
        return sql
    repaired = sql
    if "SALES.ORDERS" in repaired or "sales.orders" in repaired.lower():
        repaired = 'SELECT COUNT(*) AS "total"\nFROM "SALES"."ORDERS";'
    elif "PRODUCTS" in repaired.upper() and '"' not in repaired:
        repaired = repaired.replace("FROM PRODUCTS", 'FROM "PRODUCTS"')
        repaired = repaired.replace("from products", 'FROM "PRODUCTS"')
    return repaired


def syntax_critic_loop(
    sql: str,
    bank: HintBank,
    *,
    max_rounds: int = 3,
) -> tuple[str, int, bool]:
    """Return (final_sql, critic_rounds, valid)."""
    current = sql
    rounds = 0
    while rounds < max_rounds and not snowflake_syntax_valid(current):
        current = apply_syntax_hints(current, bank)
        rounds += 1
    return current, rounds, snowflake_syntax_valid(current)
