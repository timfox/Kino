"""Future directions (Sec. 6)."""

from __future__ import annotations

from typing import Any

FUTURE_DIRECTIONS: dict[str, list[str]] = {
    "spatial_reasoning": [
        "object_level fine-grained grounding over time",
        "scene-level 3D layout and geometric consistency",
        "structured spatial CoT without architecture changes",
    ],
    "multi_video_grounding": [
        "set-based retrieval + refinement across compilations",
        "edit-aware cut/segment anchors",
        "verifiable IoU rewards under large search spaces",
    ],
    "hour_scale_memory": [
        "three-tier buffer / event / entity memory",
        "learned write-forget with evidence pointers",
        "streaming forgetting for unbounded inputs",
    ],
    "efficient_verifiable_reasoning": [
        "budgeted evidence search (correctness + alignment + compactness)",
        "uncertainty-aware extra inspection",
        "standard timestamp/box/caption evidence schemas",
    ],
    "streaming_egocentric": [
        "explicit task-state-driven memory",
        "event-triggered episodic records",
        "proactive retrieval and safe intervention timing",
    ],
}


def future_card() -> dict[str, Any]:
    return {"future_directions": FUTURE_DIRECTIONS}
