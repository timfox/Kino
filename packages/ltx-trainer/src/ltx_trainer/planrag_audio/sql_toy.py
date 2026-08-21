"""Toy merged-SQL sketch from a simplified plan (§3.1.3, Example 2 style)."""

from __future__ import annotations

from typing import Any


def merged_sql_sketch(plan: dict[str, Any]) -> str:
    """Return a deterministic SQL-like string for smoke tests (not executable)."""
    streams = plan.get("streams", [])
    filters = plan.get("filters", {})
    fusion = plan.get("fusion", {})
    out_fields = plan.get("output", {}).get("return_fields", [])
    parts = ["WITH"]
    ctes = []
    for i, s in enumerate(streams):
        fl = filters.get(s, "")
        ctes.append(f"  s{i} AS (SELECT * FROM {s} WHERE /* filter: {fl} */ TRUE)")
    parts.append(",\n".join(ctes))
    anchor = fusion.get("anchor", "transcript")
    fields = ", ".join(out_fields) if out_fields else "*"
    parts.append(f"SELECT {fields} FROM s0 /* join on temporal_overlap, anchor={anchor} */ ;")
    return "\n".join(parts)
