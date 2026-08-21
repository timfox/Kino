"""Paper card and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.flood_physics.benchmarks import hydro_metrics_card, summary_anchors, table_2_extent
from ltx_trainer.flood_physics.constants import (
    INPUT_MODALITIES,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    TRAIN_DEFAULTS,
)
from ltx_trainer.flood_physics.losses import loss_components_card
from ltx_trainer.flood_physics.swe import swe_card


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "venue": PAPER_VENUE,
        "authors": "Tewodros Syum Gebre, Jagrati Talreja, Leila Hashemi-Beni",
        "affiliation": "NC A&T State University; UNU-INWEH",
        "findings": [
            "Hybrid UNet+FNO with SWE physics residuals on SAR/optical/DEM",
            "Flood extent IoU 0.82 ± 0.03, F1 0.90 ± 0.02 (Table II)",
            "Depth RMSE 0.21 m, velocity RMSE 0.15 m/s vs HEC-RAS reference",
            "Mass imbalance below 2.1%; physics ablation +18% depth RMSE",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "inputs": list(INPUT_MODALITIES),
        "train_defaults": dict(TRAIN_DEFAULTS),
        "swe": swe_card(),
        "losses": loss_components_card(),
        "table_2": table_2_extent(),
        "hydro_metrics": hydro_metrics_card(),
        "summary": summary_anchors(),
    }
