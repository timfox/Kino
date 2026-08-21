"""Statistical significance excerpts — Section IV-A (Wilcoxon + Bonferroni)."""

from __future__ import annotations

from typing import Any


def significance_table() -> list[dict[str, Any]]:
    """Paper-reported paired Wilcoxon tests (per-fold unit; Bonferroni corrected)."""
    return [
        {
            "comparison": "KUL BI=0 vs BI=1 (LOTO)",
            "loss": "PCC",
            "p_value": "< 0.01",
            "significant": True,
        },
        {
            "comparison": "KUL BI=0 vs BI=1 (LOTO)",
            "loss": "PCC_delta",
            "p_value": "< 0.01",
            "significant": True,
        },
        {
            "comparison": "NJU cEEGrid BI=0.185 vs BI=1 (LOTO)",
            "loss": "PCC",
            "p_value": "< 0.001",
            "significant": True,
        },
        {
            "comparison": "DTU BI=0.056 vs BI=1 (LOTO)",
            "loss": "PCC",
            "p_value": "0.12",
            "significant": False,
            "note": "Limited pair repetition (60 unique stimuli, 1:1 trial ratio).",
        },
        {
            "comparison": "DTU BI=0.056 vs BI=1 (LOTO)",
            "loss": "PCC_delta",
            "p_value": "0.08",
            "significant": False,
        },
    ]


def lopeo_mitigation_summary() -> dict[str, Any]:
    """Headline: LOPEO collapses KUL accuracy spread across BI settings."""
    return {
        "KUL_LOTO_acc_range": (0.6493, 0.8319),
        "KUL_LOPEO_acc_range": (0.6420, 0.6467),
        "interpretation": (
            "LOPEO removes stimulus-pair leakage so BI-induced inflation disappears; "
            "LOEO on NJU reduces accuracy but yields OOD-stimulus evaluation."
        ),
    }
