"""Paper tables (Park et al., arXiv:2504.14396)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spherediff.config import (
    NUM_SPHERICAL_LATENTS,
    NUM_VIEW_DIRECTIONS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
)

# Table 1 — user study (% preference, higher better)
TABLE1_USER_STUDY = {
    "static": {
        "360 LoRA": {"distortion": 21.43, "end_continuity": 23.81, "image_quality": 21.43, "text": 20.24},
        "Text2Light": {"distortion": 5.95, "end_continuity": 4.76, "image_quality": 8.33, "text": 5.95},
        "PanFusion": {"distortion": 14.29, "end_continuity": 10.71, "image_quality": 10.71, "text": 16.67},
        "DynamicScaler": {"distortion": 20.24, "end_continuity": 25.00, "image_quality": 25.00, "text": 27.38},
        "SphereDiff": {"distortion": 38.10, "end_continuity": 35.71, "image_quality": 34.52, "text": 29.76},
    },
    "live": {
        "360 LoRA+AnimateDiff": {"distortion": 25.00, "motion": 27.38, "flicker": 23.81},
        "360DVD": {"distortion": 11.90, "motion": 8.33, "flicker": 14.29},
        "DynamicScaler": {"distortion": 30.95, "motion": 28.57, "flicker": 29.76},
        "SphereDiff": {"distortion": 32.14, "motion": 35.71, "flicker": 32.14},
    },
}

# Table 2 — automated metrics (1–5 scale unless noted)
TABLE2_AUTOMATED = {
    "static": {
        "DynamicScaler": {"distortion": 2.854, "end_continuity": 3.985, "image_quality": 4.496, "clip": 0.2750},
        "SphereDiff": {"distortion": 3.238, "end_continuity": 4.892, "image_quality": 4.496, "clip": 0.5875},
    },
    "live": {
        "DynamicScaler": {"distortion": 1.971, "motion_smooth": 0.9943, "temporal_flicker": 0.9918},
        "SphereDiff": {"distortion": 2.579, "motion_smooth": 0.9956, "temporal_flicker": 0.9941},
    },
}

# Table 3 — ablation static (SANA)
TABLE3_ABLATION = {
    "Nearest sampling": {"distortion": 2.039, "end_continuity": 3.400, "clip": 0.3250},
    "Nearest + Weighted Avg.": {"distortion": 2.829, "end_continuity": 4.625, "clip": 0.5125},
    "Dynamic sampling": {"distortion": 2.421, "end_continuity": 4.086, "clip": 0.1375},
    "Dynamic + Weighted Avg.": {"distortion": 3.238, "end_continuity": 4.892, "clip": 0.5875},
}

TABLE4_METHODS = {
    "360 LoRA": {"latent": "ERP", "tuning_free": False, "wallpaper": "static"},
    "DynamicScaler": {"latent": "ERP", "tuning_free": True, "wallpaper": "static+live"},
    "SphereDiff": {"latent": "Spherical", "tuning_free": True, "wallpaper": "static+live"},
}


def ours_beats_dynamic_scaler(metric: str = "distortion") -> bool:
    s = TABLE2_AUTOMATED["static"]["SphereDiff"][metric]
    d = TABLE2_AUTOMATED["static"]["DynamicScaler"][metric]
    return s > d


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "num_latents": NUM_SPHERICAL_LATENTS,
        "num_views": NUM_VIEW_DIRECTIONS,
        "table1_user": TABLE1_USER_STUDY,
        "table2_auto": TABLE2_AUTOMATED,
        "table3_ablation": TABLE3_ABLATION,
        "table4_methods": TABLE4_METHODS,
    }
