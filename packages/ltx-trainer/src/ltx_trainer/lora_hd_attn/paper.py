"""Paper card and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lora_hd_attn.attention import attention_card
from ltx_trainer.lora_hd_attn.benchmarks import fig1_curve_rows, summary_anchors
from ltx_trainer.lora_hd_attn.constants import (
    ASYMPTOTIC,
    PAPER_ARXIV,
    PAPER_CODE,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    RESULTS,
    TRAIN_DEFAULTS,
)
from ltx_trainer.lora_hd_attn.references import REFERENCES


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "venue": PAPER_VENUE,
        "code": PAPER_CODE,
        "authors": "O. Duranthon, F. Boncoraglio, L. Zdeborová (EPFL)",
        "findings": [
            "Sharp HD asymptotics for rank-one LoRA after extensive-rank attention pre-training",
            "Effective noise Delta_eff = Delta/2 + Q0 - 2M + Q summarizes pre-training impact",
            "Reused-sequence regime: test error vs reconstruction overlap can misalign",
            "Active fine-tuning by low-variance frozen pre-activations reduces Delta_eff",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    from ltx_trainer.lora_hd_attn.pipeline import run_active_ft_demo, run_demo, run_reused_sequences_demo

    return {
        "paper": paper_card(),
        "architecture": attention_card(),
        "asymptotic": ASYMPTOTIC,
        "results": list(RESULTS),
        "train_defaults": dict(TRAIN_DEFAULTS),
        "fig1_rows": fig1_curve_rows(),
        "summary": summary_anchors(),
        "demo": run_demo(),
        "active_ft": run_active_ft_demo(),
        "reused_sequences": run_reused_sequences_demo(),
        "references": list(REFERENCES),
    }
