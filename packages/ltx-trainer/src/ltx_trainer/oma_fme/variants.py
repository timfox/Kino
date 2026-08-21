"""Framework variants CRC / PRA × ERP / CMP (Table 1, Sec. 3)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class SharingStrategy(str, Enum):
    CRC = "CRC"  # cascaded HD→4K→8K
    PRA = "PRA"  # per-resolution anchors


class ProjectionFormat(str, Enum):
    ERP = "ERP"
    CMP = "CMP"


def variant_name(projection: ProjectionFormat, strategy: SharingStrategy) -> str:
    return f"{projection.value}-{strategy.value}"


FRAMEWORK_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "ERP-CRC",
        "projection": "ERP",
        "strategy": "CRC",
        "tiles": 1,
        "flow": "HD→4K→8K cascade, single ERP tile",
    },
    {
        "name": "ERP-PRA",
        "projection": "ERP",
        "strategy": "PRA",
        "tiles": 1,
        "flow": "Per-resolution anchor (LQ/MQ/HQ) on ERP",
    },
    {
        "name": "CMP-CRC",
        "projection": "CMP",
        "strategy": "CRC",
        "tiles": 6,
        "flow": "HD→4K→8K cascade per cubemap face",
    },
    {
        "name": "CMP-PRA",
        "projection": "CMP",
        "strategy": "PRA",
        "tiles": 6,
        "flow": "Per-face, per-resolution anchors",
    },
]

CMP_FACE_NAMES = ("front", "back", "left", "right", "top", "bottom")
