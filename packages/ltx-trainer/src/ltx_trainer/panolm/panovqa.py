"""PanoVQA dataset card and QA taxonomy (Sec. III-A, Table 2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panolm.config import (
    NUM_QA_TYPES,
    PANOVQA_FRAMES,
    PANOVQA_MINI_TRAIN,
    PANOVQA_MINI_VAL,
    PANOVQA_TOTAL_QA,
    PANOVQA_TRAIN,
    PANOVQA_VAL,
)

# Table 2 — QA type counts
QA_TYPES: list[dict[str, Any]] = [
    {"id": "N1", "split": "Normal", "task": "Scene Captioning", "samples": 118_101},
    {"id": "N2", "split": "Normal", "task": "Object Identification", "samples": 144_927},
    {"id": "N3", "split": "Normal", "task": "O2O Spatial Relationships", "samples": 117_174},
    {"id": "N4", "split": "Normal", "task": "E2O Spatial Relationships", "samples": 128_793},
    {"id": "O1", "split": "Occluded", "task": "Occlusion Relationships", "samples": 698},
    {"id": "O2", "split": "Occluded", "task": "Actions of Occluded Objects", "samples": 252},
    {"id": "O3", "split": "Occluded", "task": "Actions to Prevent Accidents", "samples": 342},
    {"id": "D1", "split": "Accident", "task": "Environment/Weather", "samples": 38_619},
    {"id": "D2", "split": "Accident", "task": "Collision Risk Assessment", "samples": 28_774},
    {"id": "D3", "split": "Accident", "task": "Severity Assessment", "samples": 21_270},
    {"id": "D4", "split": "Accident", "task": "Action Avoidance", "samples": 29_405},
    {"id": "D5", "split": "Accident", "task": "Time2Collision Estimation", "samples": 25_559},
]

OBJECT_TUPLE_FIELDS = ("category", "direction", "distance", "visibility")
ACCIDENT_TUPLE_FIELDS = ("category", "direction", "distance", "speed")


def dataset_card() -> dict[str, Any]:
    return {
        "name": "PanoVQA",
        "total_qa": PANOVQA_TOTAL_QA,
        "train": PANOVQA_TRAIN,
        "val": PANOVQA_VAL,
        "mini": {"train": PANOVQA_MINI_TRAIN, "val": PANOVQA_MINI_VAL},
        "frames": PANOVQA_FRAMES,
        "subsets": ["PanoVQA-N", "PanoVQA-O", "PanoVQA-D"],
        "sources": {
            "N": "NuScenes",
            "O": "BlendPASS",
            "D": "DeepAccident",
        },
        "qa_types": QA_TYPES,
        "annotation_format": "quadruple (category, direction, distance, visibility|speed)",
    }
