"""Prompt-sophistication rubric L1–L4 (Appendix A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PromptLevel = Literal["L1", "L2", "L3", "L4"]


@dataclass(frozen=True)
class PromptRubricLevel:
    level: PromptLevel
    name: str
    criteria: str


def rubric_levels() -> list[PromptRubricLevel]:
    return [
        PromptRubricLevel("L1", "Basic task articulation", "Single instruction, no context files"),
        PromptRubricLevel("L2", "Initial structure", "Role or list or one context file"),
        PromptRubricLevel("L3", "Schema-enforced structure", "2+ context files with distinct roles + format/rubric"),
        PromptRubricLevel(
            "L4",
            "Programmatic + infrastructure-integrated",
            "Automated pipeline, schema-validated, cross-tool handoff",
        ),
    ]


def classify_prompt_level(
    *,
    context_files: int,
    has_schema: bool,
    automated: bool,
    has_role_or_structure: bool,
) -> PromptLevel:
    """Toy classifier mirroring Appendix A ordering."""
    if automated and has_schema:
        return "L4"
    if context_files >= 2 and has_schema:
        return "L3"
    if has_role_or_structure or context_files >= 1:
        return "L2"
    return "L1"
