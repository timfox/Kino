"""Four-probe audit protocol — Table 1."""

from __future__ import annotations

from typing import Any


def table_i_probes() -> list[dict[str, Any]]:
    return [
        {
            "probe": "A",
            "question": "Subject-disjoint LOSO performance on E-DAIC",
            "protocol": "LOSO; macro-F1, AUROC, AP",
            "primary_outputs": ["macro_f1", "auroc", "ap"],
        },
        {
            "probe": "B",
            "question": "Official-split ranking instability",
            "protocol": "96 configs (8 bundles × 2 poolers × 6 learners); bootstrap",
            "primary_outputs": ["rank_overlap", "p_rank_1", "cv_test_correlation"],
        },
        {
            "probe": "C",
            "question": "Zero-shot external validation",
            "protocol": "CMDC text + ANDROIDS pipelines on MODMA/PDCH",
            "primary_outputs": ["macro_f1", "auroc"],
        },
        {
            "probe": "D",
            "question": "SRDS symptom-density sensitivity (text vs audio)",
            "protocol": "Paired heavy vs neutral slices; 5 seeds",
            "primary_outputs": ["delta_heavy_minus_neutral", "text_minus_audio_gap"],
        },
    ]
