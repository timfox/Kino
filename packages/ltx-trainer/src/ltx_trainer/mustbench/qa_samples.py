"""Built-in MUSTBENCH QA fixtures for eval smoke."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QASample:
    task: str
    song_id: str
    question: str
    gold: Any
    pred: Any | None = None


def builtin_qa_samples() -> list[QASample]:
    return [
        QASample("TSG", "song-01", "When do drums enter?", 45.0, 46.5),
        QASample("TSG", "song-02", "When does vocals exit?", 120.0, 118.0),
        QASample("LTR", "song-03", "Transition at 60s?", "A", "A"),
        QASample("LTR", "song-04", "Transition at 90s?", "C", "B"),
        QASample("GTO", "song-05", "Order of three events", "ABC", "ABC"),
        QASample("MTR", "song-06", "Highest arousal interval", [(70.0, 110.0)], [(72.0, 108.0)]),
        QASample("TAD", "song-07", "Describe change at 30s", "drums surge forward", "the drums surge with punchier attack"),
    ]


def samples_by_task(samples: list[QASample]) -> dict[str, list[QASample]]:
    out: dict[str, list[QASample]] = {}
    for s in samples:
        out.setdefault(s.task, []).append(s)
    return out
