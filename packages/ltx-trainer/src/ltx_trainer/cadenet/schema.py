"""Detections, weather labels, and track state (Sec. III)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class WeatherCondition(str, Enum):
    RAIN = "rain"
    FOG = "fog"
    SAND = "sand"
    SNOW = "snow"
    CLEAR = "clear"


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int = 0
    score: float = 0.0
    source: str = "S"

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: Detection) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: Detection
    smoothed_conf: float = 0.0
    misses: int = 0
    hits: int = 0
    kalman_state: list[float] = field(default_factory=lambda: [0.0] * 7)


DAWN_COUNTS: dict[str, int] = {"fog": 600, "rain": 200, "sand": 323, "snow": 204}
