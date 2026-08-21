"""SpecBench deficiency taxonomy (IEEE 1028-1997 + paper §2)."""

from __future__ import annotations

from enum import Enum


class DeficiencyClass(str, Enum):
    OMISSION = "omission"
    AMBIGUOUS = "ambiguous"
    INCONSISTENT = "inconsistent"
    INCORRECT = "incorrect"


class GoldTier(str, Enum):
    CORE = "core"
    EXTENDED = "extended"


DEFICIENCY_CLASSES: tuple[DeficiencyClass, ...] = (
    DeficiencyClass.OMISSION,
    DeficiencyClass.AMBIGUOUS,
    DeficiencyClass.INCONSISTENT,
    DeficiencyClass.INCORRECT,
)

REPOSITORY_LABELS: dict[str, str] = {
    "kubernetes": "Kubernetes KEPs",
    "react": "React RFCs",
    "rust": "Rust RFCs",
    "tvm": "TVM RFCs",
    "vllm": "vLLM RFC/design proposals",
}


def classify_deficiency_text(text: str) -> DeficiencyClass:
    """Heuristic deficiency class from free text (stub judge helper)."""
    lower = text.lower()
    if any(w in lower for w in ("contradict", "conflict", "incorrect", "wrong", "violat")):
        return DeficiencyClass.INCORRECT
    if any(w in lower for w in ("inconsistent", "conflict with", "contradict")):
        return DeficiencyClass.INCORRECT
    if any(w in lower for w in ("ambiguous", "unclear", "underspecified", "not defined", "tbd")):
        return DeficiencyClass.AMBIGUOUS
    if any(w in lower for w in ("missing", "not described", "not specified", "lacks", "no ")):
        return DeficiencyClass.OMISSION
    return DeficiencyClass.INCONSISTENT
