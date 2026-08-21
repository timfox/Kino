"""31-scenario library LoC comparison (Table 1, Sec. 4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.modular2simple.config import (
    LIBRARY_MODULAR_LOC,
    LIBRARY_REDUCTION_PCT,
    LIBRARY_SCENARIO_COUNT,
    LIBRARY_TRADITIONAL_LOC,
    SIMPLE_REUSE_LOC,
)

# Table 1 per-scenario LoC (modular vs traditional)
TABLE1_LOC: list[dict[str, int | float]] = [
    {"scenario": i, "modular_loc": m, "traditional_loc": t, "reduction_pct": round(100 * (1 - m / t), 1)}
    for i, m, t in [
        (1, 200, 358), (2, 274, 538), (3, 205, 417), (4, 290, 608), (5, 364, 788),
        (6, 278, 596), (7, 200, 358), (8, 286, 550), (9, 360, 730), (10, 290, 608),
        (11, 376, 800), (12, 450, 980), (13, 364, 788), (14, 274, 538), (15, 188, 346),
        (16, 200, 358), (17, 286, 550), (18, 360, 730), (19, 290, 608), (20, 376, 800),
        (21, 450, 980), (22, 364, 788), (23, 286, 550), (24, 372, 742), (25, 446, 922),
        (26, 376, 800), (27, 462, 992), (28, 536, 1172), (29, 450, 980), (30, 360, 730),
        (31, 274, 538),
    ]
]


def total_reduction_pct(modular: int, traditional: int) -> float:
    return round(100.0 * (1.0 - modular / traditional), 1)


def library_summary() -> dict[str, Any]:
    mod_sum = sum(r["modular_loc"] for r in TABLE1_LOC)
    trad_sum = sum(r["traditional_loc"] for r in TABLE1_LOC)
    return {
        "scenario_count": LIBRARY_SCENARIO_COUNT,
        "modular_loc_total": mod_sum,
        "traditional_loc_total": trad_sum,
        "table_total_modular": LIBRARY_MODULAR_LOC,
        "table_total_traditional": LIBRARY_TRADITIONAL_LOC,
        "reduction_pct": LIBRARY_REDUCTION_PCT,
        "simple_reuse_loc": list(SIMPLE_REUSE_LOC),
        "behavior_templates": 2,
    }


def library_demo() -> dict[str, Any]:
    s = library_summary()
    return {
        "scenarios": s["scenario_count"],
        "reduction_pct": s["reduction_pct"],
        "modular_loc": s["table_total_modular"],
        "traditional_loc": s["table_total_traditional"],
    }
