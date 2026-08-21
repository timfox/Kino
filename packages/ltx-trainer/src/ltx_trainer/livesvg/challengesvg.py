"""ChallengeSVG benchmark metadata (Sec. 4.1, suppl. A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChallengeSVGRecord:
    example_id: str
    source: str = "SVGX-Core-250k"
    multi_object: bool = True
    has_background: bool = True


def challengesvg_manifest() -> dict[str, Any]:
    """Benchmark card — 35 complex multi-object SVGs (paper)."""
    return {
        "name": "ChallengeSVG",
        "num_examples": 35,
        "source_hub": "SVGX-Core-250k",
        "stress_factors": [
            "multi_object_scenes",
            "non_empty_backgrounds",
            "dense_path_color_structure",
            "non_skeleton_subjects",
        ],
        "paired_with": "AniClipart (43 examples)",
        "evaluation_generators": ["LiveSVG (WAN 2.2)"],
    }


def example_records_preview(n: int = 4) -> list[dict[str, str]]:
    """Synthetic preview rows for agents (no bundled SVG assets)."""
    return [
        {"example_id": f"challengesvg_{i:03d}", "source": "SVGX-Core-250k", "note": "preview stub"}
        for i in range(n)
    ]
