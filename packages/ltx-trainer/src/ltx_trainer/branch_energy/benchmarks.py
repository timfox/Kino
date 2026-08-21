"""Table I and summary anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.branch_energy.constants import TEST_CASES


def table_i_rows() -> list[dict[str, str]]:
    """Qualitative Table I — branch localization vs classical theories."""
    return [
        {
            "case": "IV-A Balanced RL",
            "ieee_1459": "Per-phase p1459 phase-shifted 2θ from branch form",
            "akagi_pq": "q̄ ≠ 0; per-phase profile not exposed",
            "cpc": "Reactive current global; no per-phase storage",
            "fbd": "No per-phase split",
            "branch_loc": "Per-phase W'_L,x recovered; 2θ shift exposed",
        },
        {
            "case": "IV-B Open phase",
            "ieee_1459": "S > P from asymmetry",
            "akagi_pq": "qαβ ≠ 0 for purely resistive load",
            "cpc": "Active currents in all 3 phases incl. open",
            "fbd": "Nonactive current present",
            "branch_loc": "W'_L = W'_C ≡ 0; all Joule in Rab",
        },
        {
            "case": "IV-C TRIAC switched",
            "ieee_1459": "Large D, QF from harmonics",
            "akagi_pq": "qαβ ≠ 0 despite no storage",
            "cpc": "Scattered/generated currents",
            "fbd": "Void/nonactive populates",
            "branch_loc": "W'_L = W'_C ≡ 0; G(t) captures switching",
        },
        {
            "case": "IV-D Topology indeterminacy",
            "ieee_1459": "Single S regardless of topology",
            "akagi_pq": "Single p–q pair",
            "cpc": "Single decomposition",
            "fbd": "Single nonactive current",
            "branch_loc": "Two distinct localizations (Cor. 1)",
        },
        {
            "case": "IV-E Fluctuating phase",
            "ieee_1459": "Q1 → 0; storage misclassified",
            "akagi_pq": "q̄αβ → 0; storage missed",
            "cpc": "Fundamental reactive small",
            "fbd": "Nonactive small in fundamental",
            "branch_loc": "Time-varying W'_L or W'_C recovered",
        },
        {
            "case": "IV-F 4-wire nonlinear",
            "ieee_1459": "Per-phase S; no hysteresis split",
            "akagi_pq": "Aggregates; switching not localized",
            "cpc": "No storage assigned to hysteresis",
            "fbd": "Nonactive without locating loss",
            "branch_loc": "Per-phase pR, W'_L, W'_C; hysteresis in phase a",
        },
    ]


def summary_anchors() -> dict[str, Any]:
    return {
        "num_test_cases": len(TEST_CASES),
        "table_i_rows": len(table_i_rows()),
        "theorems": 3,
        "corollaries": 2,
    }
