"""Benchmark dataset cards (Sec. IV)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cross360.config import PAPER_ARXIV

DATASETS: dict[str, dict[str, Any]] = {
    "Matterport3D": {
        "id": "M3D",
        "type": "real",
        "fov": "incomplete_vertical",
        "tp_patches": 20,
        "tp_latitudes_deg": [-31.2, 0.0, 31.2],
        "eval_mask": "dataset_gt_valid_pixels",
    },
    "Stanford2D3D": {
        "id": "S2D3D",
        "type": "real",
        "fov": "incomplete_vertical",
        "tp_patches": 20,
        "loss": "reverse_huber",
        "note": "small training set",
    },
    "Structured3D": {
        "id": "Struct3D",
        "type": "synthetic",
        "fov": "complete_360",
        "tp_patches": 26,
        "max_depth_m": 10.0,
    },
    "3D60": {
        "id": "3D60",
        "type": "synthetic",
        "fov": "complete_360",
        "tp_patches": 26,
        "alias": "OmniDepth",
    },
}


def dataset_card(name: str) -> dict[str, Any]:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    return {"name": name, "paper": f"arXiv:{PAPER_ARXIV}", **DATASETS[name]}


def all_datasets_card() -> dict[str, Any]:
    return {"datasets": list(DATASETS.keys()), "entries": DATASETS}
