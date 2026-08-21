"""U-FIRE benchmark card — Table I scale (Sec. III)."""

from __future__ import annotations

from typing import Any


def ufire_scale_summary() -> dict[str, Any]:
    """Triplet counts and task/dataset coverage."""
    return {
        "unified_triplets_train": 325_094,
        "unified_triplets_val": 36_187,
        "unified_triplets_test": 46_759,
        "num_source_datasets": 15,
        "trainable_tasks": 9,
        "evaluation_only_ood_tasks": 2,
        "ood_task_names": [
            "Street+Modification Text→Shop",
            "Image(s)+Text→Compatible Item",
        ],
        "instruction_templates_per_task": 4,
    }


def ufire_task_rubric() -> list[dict[str, str | int]]:
    """High-level task index → name (Table I)."""
    return [
        {"id": 1, "name": "Text→Image", "trainable": True},
        {"id": 2, "name": "Sketch→Image", "trainable": True},
        {"id": 3, "name": "Street→Shop", "trainable": True},
        {"id": 4, "name": "In-Shop", "trainable": True},
        {"id": 5, "name": "Video→Image", "trainable": True},
        {"id": 6, "name": "Sketch+Text→Image", "trainable": True},
        {"id": 7, "name": "Image+Modification Text→Image", "trainable": True},
        {"id": 8, "name": "Image(s)→Compatible Item", "trainable": True},
        {"id": 9, "name": "Image+Attribute→Image", "trainable": True},
        {"id": 10, "name": "Street+Modification Text→Shop", "trainable": False},
        {"id": 11, "name": "Image(s)+Text→Compatible Item", "trainable": False},
    ]
