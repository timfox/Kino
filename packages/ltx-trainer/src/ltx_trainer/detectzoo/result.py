"""Standardized detection outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Label = Literal["human", "ai"]


@dataclass
class DetectionResult:
    """Unified predict() return type (Section 3.1)."""

    score: float
    label: Label
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
