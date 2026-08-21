"""WeatherProof semantic classes (Sec. 3)."""

from __future__ import annotations

WEATHERPROOF_CLASSES: tuple[str, ...] = (
    "background",
    "tree",
    "structure",
    "road",
    "terrain-snow",
    "terrain-grass",
    "terrain-other",
    "stone",
    "building",
    "sky",
)

NUM_CLASSES = len(WEATHERPROOF_CLASSES)
IGNORE_LABEL = 255
