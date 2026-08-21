"""Mitigation strategies (Sec. 3.2, Fig. 4)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class MitigationStrategy(str, Enum):
    DISTORTION_AWARE = "distortion_aware"
    PROJECTION_DRIVEN = "projection_driven"
    PHYSICS_GEOMETRY = "physics_geometry"
    OTHER = "other"


# Task codes from Fig. 4 legend (abbreviated)
TASK_CODES = (
    "SR",
    "IR",
    "IQA",
    "Seg",
    "Det",
    "SP",
    "LD",
    "Decom",
    "LE",
    "DE",
    "MMF",
    "T2I/V",
    "IC",
    "NVS",
)

# Preferred primary strategy per task family (Sec. 3.2 applicability)
TASK_STRATEGY_HINT: dict[str, MitigationStrategy] = {
    "SR": MitigationStrategy.DISTORTION_AWARE,
    "IR": MitigationStrategy.DISTORTION_AWARE,
    "IQA": MitigationStrategy.DISTORTION_AWARE,
    "Seg": MitigationStrategy.DISTORTION_AWARE,
    "Det": MitigationStrategy.PROJECTION_DRIVEN,
    "SP": MitigationStrategy.PROJECTION_DRIVEN,
    "LD": MitigationStrategy.PROJECTION_DRIVEN,
    "Decom": MitigationStrategy.PHYSICS_GEOMETRY,
    "LE": MitigationStrategy.PHYSICS_GEOMETRY,
    "DE": MitigationStrategy.PROJECTION_DRIVEN,
    "MMF": MitigationStrategy.PROJECTION_DRIVEN,
    "T2I/V": MitigationStrategy.DISTORTION_AWARE,
    "IC": MitigationStrategy.DISTORTION_AWARE,
    "NVS": MitigationStrategy.PROJECTION_DRIVEN,
}

REPRESENTATIVE_METHODS: dict[MitigationStrategy, list[str]] = {
    MitigationStrategy.DISTORTION_AWARE: [
        "OSRT",
        "360-SISR",
        "LAUNet",
        "DensePASS",
        "PanoFormer",
        "SphereDiffusion",
    ],
    MitigationStrategy.PROJECTION_DRIVEN: [
        "BiFuse",
        "UniFuse",
        "SphereSR",
        "Multi-Projection YOLO",
        "VideoPanda",
        "PanSplat",
    ],
    MitigationStrategy.PHYSICS_GEOMETRY: [
        "HorizonNet",
        "PhyIR",
        "PAR2Net",
        "EnvMapNet",
    ],
    MitigationStrategy.OTHER: [
        "Diffusion360",
        "PanoFree",
        "MC360IQA",
        "GoodSAM++",
    ],
}


def classify_method(name: str) -> MitigationStrategy:
    key = name.lower().replace(" ", "").replace("-", "")
    for strat, methods in REPRESENTATIVE_METHODS.items():
        for m in methods:
            if m.lower().replace(" ", "") in key or key in m.lower().replace(" ", ""):
                return strat
    if any(x in key for x in ("cubemap", "tangent", "bifuse", "unifuse", "projection")):
        return MitigationStrategy.PROJECTION_DRIVEN
    if any(x in key for x in ("horizon", "layout", "physics", "lighting")):
        return MitigationStrategy.PHYSICS_GEOMETRY
    if any(x in key for x in ("diffusion", "sam", "vq", "gan")):
        return MitigationStrategy.OTHER
    return MitigationStrategy.DISTORTION_AWARE


def strategies_card() -> dict[str, Any]:
    return {
        "strategies": [s.value for s in MitigationStrategy],
        "task_codes": list(TASK_CODES),
        "task_strategy_hints": {k: v.value for k, v in TASK_STRATEGY_HINT.items()},
        "representative_methods": {s.value: REPRESENTATIVE_METHODS[s] for s in MitigationStrategy},
    }
