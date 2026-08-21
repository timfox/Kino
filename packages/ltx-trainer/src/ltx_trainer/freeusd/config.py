"""FreeUSD — OpenUSD text as the spatial lock for TV shot composition."""

from __future__ import annotations

from dataclasses import dataclass, field


ARXIV = "openusd-spatial-llm"
PAPER = "FreeUSD"
FORMAT = "usda-1.0"


@dataclass(frozen=True)
class FreeUSDConfig:
    """ASCII USDA 1.0 scene graph for CID shot composition (no pxr required)."""

    format: str = FORMAT
    meters_per_unit: float = 1.0
    up_axis: str = "Y"
    fps: float = 24.0
    frames_default: int = 121
    expected_cast: int = 2
    camera_lens_mm: float = 35.0
    lighting: str = "high-key"
    shot_size: str = "MS two-shot"


@dataclass(frozen=True)
class SpatialPose:
    name: str
    prim: str
    translate: tuple[float, float, float]
    pose: str
    notes: str = ""


DEFAULT_TWO_SHOT: tuple[SpatialPose, ...] = (
    SpatialPose("Elena Vale", "Char_Elena", (-0.65, 0.0, 2.4), "sitting", "camera-left beige sofa"),
    SpatialPose("Julian Hart", "Char_Julian", (0.85, 0.0, 3.6), "standing", "camera-right doorway"),
)

SET_PRIMS: tuple[SpatialPose, ...] = (
    SpatialPose("beige sofa", "Set_Sofa", (-0.4, 0.0, 2.6), "set", "living-room practical"),
    SpatialPose("large windows", "Set_Windows", (0.0, 1.8, 5.5), "set", "daylight key"),
    SpatialPose("doorway", "Set_Doorway", (1.1, 0.0, 4.2), "set", "Julian blocking mark"),
)
