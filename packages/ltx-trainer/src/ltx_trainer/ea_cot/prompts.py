"""EA-CoT and control prompt templates (§3.2)."""

from __future__ import annotations

EA_COT_STAGES: tuple[str, ...] = (
    "Entity enumeration: list all people mentioned.",
    "Claim recording: write each statement linking a person to a property.",
    "Step-by-step reasoning: resolve each claim in sequence.",
    "Answer extraction: produce the answer in the required format.",
)

GENERIC_COT = "Let us think step by step."

CONTROL_PROMPTS: dict[str, str] = {
    "hyperbaton": "List and classify adjectives by type, then answer.",
    "navigate": "Track coordinate changes from the origin step by step.",
    "sports understanding": "Identify the sport and evaluate action plausibility.",
    "web of lies": EA_COT_STAGES[0],  # entity track uses full EA-CoT
}


def ea_cot_prompt(task: str = "web of lies") -> str:
    """Full Entity-Aware CoT prefix for entity-tracking tasks."""
    if task != "web of lies":
        return CONTROL_PROMPTS.get(task, GENERIC_COT)
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(EA_COT_STAGES))
