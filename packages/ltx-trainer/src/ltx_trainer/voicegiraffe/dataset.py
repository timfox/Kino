"""VOICEGIRAFFE QA item schema and built-in evaluation fixtures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class QAItem:
    id: str
    recording_id: str
    tier: str  # single_hop | multi_hop
    task: str
    domain: str
    language: str
    question: str
    choices: dict[str, str]
    gold: str
    duration_min: float
    timestamp_s: float | None = None


def _builtin_items() -> list[QAItem]:
    return [
        QAItem(
            id="vg-001",
            recording_id="rec-esports-01",
            tier="single_hop",
            task="temporal_localization",
            domain="esports",
            language="EN",
            question="When did Team Liquid ace Suning?",
            choices={"A": "[2545, 2575]", "B": "[1150, 1180]", "C": "[2580, 2610]", "D": "[2100, 2130]"},
            gold="C",
            duration_min=52.0,
            timestamp_s=2580.0,
        ),
        QAItem(
            id="vg-002",
            recording_id="rec-podcast-07",
            tier="single_hop",
            task="semantic_content",
            domain="podcast",
            language="EN",
            question="Which strategy for tunnel traffic is incorrect?",
            choices={
                "A": "Waiving specific parking fees.",
                "B": "Opening emergency lanes.",
                "C": "Early off-peak arrivals.",
                "D": "Automating parking logs via smart vehicles",
            },
            gold="D",
            duration_min=58.0,
        ),
        QAItem(
            id="vg-003",
            recording_id="rec-drama-03",
            tier="single_hop",
            task="paralinguistic",
            domain="tv_drama",
            language="EN",
            question="How does GAO sound when calling out for fish?",
            choices={
                "A": "Sharp and shaky — panic",
                "B": "Warm and lifted — dinner invite",
                "C": "Seething with rage",
                "D": "Flat and cold",
            },
            gold="B",
            duration_min=55.0,
        ),
        QAItem(
            id="vg-004",
            recording_id="rec-news-12",
            tier="multi_hop",
            task="causal_alignment",
            domain="news",
            language="EN",
            question="Which event chain triggered defensive open-sourcing?",
            choices={"A": "[1]→[6]→[4]", "B": "[7]→[3]→[2]", "C": "[6]→[1]→[4]", "D": "[4]→[5]→[6]"},
            gold="C",
            duration_min=61.0,
        ),
        QAItem(
            id="vg-005",
            recording_id="rec-sports-09",
            tier="multi_hop",
            task="event_tracking",
            domain="sports",
            language="ZH",
            question="How many times did the coach substitute the pitcher?",
            choices={"A": "0", "B": "1", "C": "2", "D": "3"},
            gold="B",
            duration_min=54.0,
        ),
        QAItem(
            id="vg-006",
            recording_id="rec-esports-01",
            tier="single_hop",
            task="acoustic_event",
            domain="esports",
            language="EN",
            question="What sudden sound interrupts the monologue?",
            choices={
                "A": "Heavy object hitting ground",
                "B": "Teacup shattering",
                "C": "Electronic beeping",
                "D": "Slapping sound",
            },
            gold="A",
            duration_min=52.0,
        ),
    ]


def load_qa_items(path: str | Path | None = None) -> list[QAItem]:
    if path is None:
        return _builtin_items()
    p = Path(path)
    rows = json.loads(p.read_text(encoding="utf-8"))
    return [QAItem(**row) for row in rows]


def iter_by_task(items: list[QAItem]) -> Iterator[tuple[str, list[QAItem]]]:
    by_task: dict[str, list[QAItem]] = {}
    for item in items:
        by_task.setdefault(item.task, []).append(item)
    for task in sorted(by_task):
        yield task, by_task[task]


def items_to_json(items: list[QAItem]) -> str:
    return json.dumps([asdict(i) for i in items], indent=2)
