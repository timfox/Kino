"""Degradation prompt templates (Appendix A.5–A.6)."""

from __future__ import annotations

from ltx_trainer.spacedg.degradations import DEGRADATION_PARAM_RANGES
from ltx_trainer.spacedg.schema import DegradationType


def degradation_aware_prompt(question: str, degradation: DegradationType | str) -> str:
    dt = DegradationType(degradation) if degradation != "original" else DegradationType.ORIGINAL
    if dt == DegradationType.ORIGINAL:
        return question
    key = dt.value
    pr = DEGRADATION_PARAM_RANGES.get(key, "see benchmark config")
    return f"{question}\nNote: The provided images have a {key} degradation with parameters range: {pr}\n"


def cot_degradation_prompt(question: str) -> str:
    labels = ", ".join(d.value for d in DegradationType if d != DegradationType.ORIGINAL)
    return (
        f"The images may exhibit at most one of the following degradations: [{labels}].\n"
        "Output format (strict order):\n"
        "1) First describe the degradation using ONE sentence, then output exactly one line: "
        "<degradation>LABEL</degradation>\n"
        "2) Then briefly think step by step about the user's visual/spatial question.\n"
        "Finally output the task answer inside <answer>...</answer>.\n"
        f"Question: {question}"
    )
