"""PanoEnv-QA taxonomy and dataset card (Table 1)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoenv.config import (
    PANENV_ENVIRONMENTS,
    PANENV_QA_TOTAL,
    PANENV_SCENES,
    TEST_SAMPLES,
)

# Table 1 — category distribution
QA_CATEGORIES: list[dict[str, Any]] = [
    {"id": "attr", "name": "Intrinsic Attribute Comparison", "count": 2_975, "pct": 20.1},
    {"id": "dist", "name": "Object Distance Estimation", "count": 2_975, "pct": 20.1},
    {"id": "pos", "name": "Relative Spatial Positioning", "count": 2_975, "pct": 20.1},
    {"id": "env", "name": "Environment Identification", "count": 2_965, "pct": 20.0},
    {"id": "view", "name": "Camera View Source Identification", "count": 2_937, "pct": 19.8},
]

QUESTION_TYPES = ("yes_no", "mcq", "distance", "spatial", "counting", "open_ended")


def dataset_card() -> dict[str, Any]:
    return {
        "name": "PanoEnv-QA",
        "total_qa": PANENV_QA_TOTAL,
        "scenes": PANENV_SCENES,
        "environments": PANENV_ENVIRONMENTS,
        "test_subset": TEST_SAMPLES,
        "base": "TartanAir",
        "modalities": ["RGB", "depth", "semantic segmentation"],
        "categories": QA_CATEGORIES,
        "question_types": list(QUESTION_TYPES),
        "annotation": "programmatic 3D GT (depth, 3D boxes, semantics)",
    }
