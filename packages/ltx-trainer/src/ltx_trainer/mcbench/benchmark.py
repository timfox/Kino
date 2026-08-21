"""Multimodal multicontext safety task definition (Sec. 2.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SafetyLabel = Literal["safe", "unsafe"]


@dataclass(frozen=True)
class MultimodalContext:
    image_id: str
    audio_id: str
    speech: str


@dataclass(frozen=True)
class ScenarioInstance:
    context: MultimodalContext
    label: SafetyLabel
    predicate: str
    category: str
    paired: bool = True


def classify_safety(label: SafetyLabel, prediction: SafetyLabel) -> bool:
    return label == prediction


def parse_predicate_premises(predicate: str) -> list[str]:
    """Extract IF-THEN premise clauses from ground-truth predicate stub."""
    upper = predicate.upper()
    if "IF" not in upper or "THEN" not in upper:
        return [predicate.strip()] if predicate.strip() else []
    body = predicate.split("THEN", 1)[0]
    body = body.replace("IF", "", 1)
    parts = [p.strip() for p in body.replace("AND", "\n").split("\n") if p.strip()]
    return parts or [predicate.strip()]
