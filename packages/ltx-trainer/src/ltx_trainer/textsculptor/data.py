"""TextSculpt-Data construction helpers (Sec. 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EditingPair:
    """Aligned source–target text editing sample."""

    source_image_id: str
    target_image_id: str
    instruction: str
    source_text: str
    target_text: str
    task_type: str


def dataset_card() -> dict[str, Any]:
    """TextSculpt-Data summary."""
    return {
        "name": "TextSculpt-Data",
        "total_m": 3.2,
        "t2i_verified_m": 1.2,
        "t2i_candidates_m": 3.0,
        "editing_pairs_m": 2.0,
        "pipelines": [
            "vlm_caption_rewrite + image_gen + paddleocr_filter",
            "corpus + python_render + compositor",
        ],
        "task_coverage": ["addition", "replacement", "removal", "hybrid"],
        "background_consistency": "pixel-exact outside edit mask (programmatic)",
    }


def bench_card() -> dict[str, Any]:
    """TextSculpt-Bench summary."""
    return {
        "name": "TextSculpt-Bench",
        "samples_per_task": 200,
        "total_samples": 800,
        "source_images": {"pexels": 188, "poster_nanobanana": 168},
        "metrics": ["text_accuracy", "visual_quality", "background_preservation"],
        "vq_criteria": ["location", "style", "physical"],
    }
