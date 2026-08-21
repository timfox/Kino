"""Seven SARL probing tasks (Sec. 3.3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTask:
    name: str
    group: str  # localization | semantic | room
    kind: str  # continuous | categorical
    bins: int
    value_range: tuple[float, float] | None = None


SOURCE_TASKS: tuple[ProbeTask, ...] = (
    ProbeTask("azimuth", "localization", "continuous", 36, (-180.0, 180.0)),
    ProbeTask("elevation", "localization", "continuous", 12, (-60.0, 60.0)),
    ProbeTask("distance", "localization", "continuous", 20, (0.5, 2.5)),
    ProbeTask("class", "semantic", "categorical", 7, None),
)

ROOM_TASKS: tuple[ProbeTask, ...] = (
    ProbeTask("rt60", "room", "continuous", 29, (0.1, 3.0)),
    ProbeTask("volume", "room", "continuous", 5, (60.0, 2500.0)),
    ProbeTask("shape", "room", "categorical", 4, None),
)

ALL_TASKS: tuple[ProbeTask, ...] = SOURCE_TASKS + ROOM_TASKS


INPUT_FORMATS: tuple[str, ...] = ("mono", "stereo", "binaural", "foa")
TRAINING_PARADIGMS: tuple[str, ...] = ("ssl", "supervised", "codec")

ENCODERS: tuple[dict[str, str], ...] = (
    {"name": "A-MAE", "input": "mono", "paradigm": "ssl"},
    {"name": "SELD-S", "input": "stereo", "paradigm": "supervised"},
    {"name": "EnCodec", "input": "stereo", "paradigm": "codec"},
    {"name": "SR-VAE", "input": "stereo", "paradigm": "codec"},
    {"name": "BANC", "input": "binaural", "paradigm": "codec"},
    {"name": "GRAM-B", "input": "binaural", "paradigm": "ssl"},
    {"name": "S-AST", "input": "binaural", "paradigm": "supervised"},
    {"name": "SFD", "input": "binaural", "paradigm": "ssl"},
    {"name": "W-JEPA", "input": "binaural", "paradigm": "ssl"},
    {"name": "AVSA", "input": "binaural", "paradigm": "ssl"},
    {"name": "EINv2", "input": "foa", "paradigm": "supervised"},
    {"name": "SELD-F", "input": "foa", "paradigm": "supervised"},
    {"name": "GRAM-F", "input": "foa", "paradigm": "ssl"},
)
