"""Paper card and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dilated_sym_diff.benchmarks import fig3_iou_curve, fig4_iou_curve, summary_anchors
from ltx_trainer.dilated_sym_diff.constants import (
    PAPER_ARXIV,
    PAPER_AUTHOR,
    PAPER_TITLE,
    PAPER_URL,
    RADIUS_RULES,
    TRAIN_DEFAULTS,
)
from ltx_trainer.dilated_sym_diff.layer import layer_card
from ltx_trainer.dilated_sym_diff.metrics import metrics_card
from ltx_trainer.dilated_sym_diff.morphology import morphology_card


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "author": PAPER_AUTHOR,
        "findings": [
            "Dilated symmetric difference A ⊕^r_△ B compensates bounded registration error δ_align",
            "Choose r > δ_align but small enough to preserve gap detection (Sec. 3 tradeoff)",
            "Fig. 1 die misalignment δ_align ≈ 7.5; r = 8 achieves IoU = 1.0",
            "Extension: learnable dilation radius in differentiable morphological NN pipeline",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "morphology": morphology_card(),
        "metrics": metrics_card(),
        "radius_rules": dict(RADIUS_RULES),
        "train_defaults": dict(TRAIN_DEFAULTS),
        "fig3_iou_curve": fig3_iou_curve(),
        "fig4_iou_curve": fig4_iou_curve(),
        "layer": layer_card(),
        "summary": summary_anchors(),
    }
