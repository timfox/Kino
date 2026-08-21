"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.s3u_sar.benchmarks import (
    dataset_card,
    summary_anchors,
    table_1_baselines,
    table_1_s3u_metrics,
    table_2_ablation,
    table_3_cross_category,
    table_4_orientation,
)
from ltx_trainer.s3u_sar.constants import (
    AIRCRAFT_CATEGORIES,
    GITHUB_URL,
    KEYPOINT_NAMES,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    TRAIN_DEFAULTS,
)
from ltx_trainer.s3u_sar.losses import loss_components_card
from ltx_trainer.s3u_sar.references import reference_anchors
from ltx_trainer.s3u_sar.structure import structure_representation_card


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "github": GITHUB_URL,
        "authors": "Yifei Yin, Xiaogang Yu, Hao Shi, Liang Chen, Wei Li",
        "affiliation": "Beijing Institute of Technology",
        "findings": [
            "Semantic Scattering Structure Understanding: S=(K,A,G) for SAR aircraft",
            "KP-SAR-Aircraft-1.0: 2990 Gaofen-3 samples, 10 keypoints, visibility + topology",
            "S³U-SAR: HRNet + physics losses + confidence-gated supervision → AP 59.3%",
            "Downstream orientation: +17.07 pp P1°, +19.31 pp P5° vs strongest baseline",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "dataset": dataset_card(),
        "categories": list(AIRCRAFT_CATEGORIES),
        "keypoints": list(KEYPOINT_NAMES),
        "structure": structure_representation_card(),
        "losses": loss_components_card(),
        "train_defaults": dict(TRAIN_DEFAULTS),
        "table_1": table_1_baselines(),
        "table_1_s3u": table_1_s3u_metrics(),
        "table_2": table_2_ablation(),
        "table_3": table_3_cross_category(),
        "table_4": table_4_orientation(),
        "summary": summary_anchors(),
        "references": reference_anchors(),
    }
