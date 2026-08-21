"""Mann-Whitney and BH-FDR bookkeeping (§III-C) — paper anchors, not re-run stats."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deopt_reopt.constants import (
    DIRECT3_DR_AHEAD,
    SINGLE_SHOT_D_WINS_BH,
    SINGLE_SHOT_DR_WINS_BH,
    TABLE2_ITERATIVE,
    TABLE2_SINGLE_SHOT,
)


def significance_marker(sig: str | None) -> str:
    if sig is None:
        return ""
    if sig == "dr_bh":
        return "†"
    if sig == "d_bh":
        return "†"
    if sig == "dr_unadj":
        return "*"
    if sig == "d_unadj":
        return "*"
    return ""


def table2_rows(workflow: str = "single_shot") -> list[dict[str, Any]]:
    raw = TABLE2_SINGLE_SHOT if workflow == "single_shot" else TABLE2_ITERATIVE
    rows: list[dict[str, Any]] = []
    for r in raw:
        row = dict(r)
        row["marker"] = significance_marker(row.get("sig"))  # type: ignore[arg-type]
        if row.get("d_med") and row.get("dr_med"):
            d_med = float(row["d_med"])  # type: ignore[arg-type]
            dr_med = float(row["dr_med"])  # type: ignore[arg-type]
            row["dr_over_d_ratio"] = round(dr_med / d_med, 2) if d_med > 0 else None
        rows.append(row)
    return rows


def single_shot_significance_summary() -> dict[str, Any]:
    return {
        "testable_cases": 18,
        "bh_fdr_significant": 8,
        "deopt_reopt_wins_bh": list(SINGLE_SHOT_DR_WINS_BH),
        "direct_wins_bh": list(SINGLE_SHOT_D_WINS_BH),
        "direct3_testable": 19,
        "direct3_dr_ahead_unadjusted": list(DIRECT3_DR_AHEAD),
        "note": "Performance tested over successful trials only; feasibility reported separately",
    }


def iterative_significance_summary() -> dict[str, Any]:
    sig_rows = [r for r in TABLE2_ITERATIVE if r.get("sig") == "dr_bh" or r.get("sig") == "d_bh"]
    dr_wins = [(r["model"], r["kernel"]) for r in sig_rows if str(r.get("sig", "")).startswith("dr")]
    d_wins = [(r["model"], r["kernel"]) for r in sig_rows if str(r.get("sig", "")).startswith("d")]
    return {
        "testable_cases": 23,
        "bh_fdr_significant": 7,
        "deopt_reopt_wins_bh": dr_wins,
        "direct_wins_bh": d_wins,
        "large_effect_q235": ("conv2d", "ddgemm", "bgemm"),
        "o120_gap_narrowed": True,
    }
