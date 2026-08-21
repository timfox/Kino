"""Figure anchors and Table-style summaries."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lora_hd_attn.constants import FIG1, FIG2, FIG3, FIG4, RESULTS


def summary_anchors() -> dict[str, Any]:
    return {
        "fig1": FIG1,
        "fig2": FIG2,
        "fig3": FIG3,
        "fig4": FIG4,
        "results": list(RESULTS),
        "claims": [
            "LoRA improves over frozen pre-training (lambda' = inf baseline)",
            "Effective noise Delta_eff summarizes pre-training impact on fine-tuning",
            "e=1 reused sequences: overlap vs test-error mismatch possible",
            "Active fine-tuning reduces Delta_eff vs uniform sampling",
        ],
    }


def fig1_curve_rows() -> list[dict[str, Any]]:
    return [
        {"lambda": lam, "role": "pretrain_regularization", "figure": "Fig.1"}
        for lam in FIG1["lambda_grid"]
    ]
