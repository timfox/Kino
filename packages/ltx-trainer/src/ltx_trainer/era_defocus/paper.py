"""Paper card and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.era_defocus.benchmarks import dataset_card, summary_anchors, table_1_results
from ltx_trainer.era_defocus.constants import (
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    TRAIN_DEFAULTS,
)
from ltx_trainer.era_defocus.losses import loss_components_card
from ltx_trainer.era_defocus.unrolling import unrolling_block_card


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "venue": PAPER_VENUE,
        "authors": "Tu Vo, Chan Y. Park",
        "affiliation": "KC Machine Learning Lab, Seoul",
        "findings": [
            "Error-aware ALM unrolling with sparse E corrects PSF estimation errors",
            "Compact kernel basis + per-pixel weights for non-Gaussian spatially varying blur",
            "SOTA PSNR/SSIM on DPDD, RealDOF, RTF; strong CUHK generalization",
            "ErA w/o E ablation confirms error term (+0.326 dB on DPDD)",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "dataset": dataset_card(),
        "train_defaults": dict(TRAIN_DEFAULTS),
        "structure": unrolling_block_card(),
        "losses": loss_components_card(),
        "table_1": table_1_results(),
        "summary": summary_anchors(),
    }
