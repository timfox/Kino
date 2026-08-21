"""FogAct action categories (Sec. 3)."""

from __future__ import annotations

FOGACT_CLASSES: tuple[str, ...] = (
    "adjust clothes",
    "adjust glasses",
    "applaud",
    "block sunlight",
    "carry heavy objects",
    "check shoe sole",
    "check watch",
    "dribble basketball",
    "drink",
    "eat",
    "fix hair",
    "hailing a taxi",
    "hand over",
    "handshakes",
    "head down walk",
    "high knee lift",
    "jump",
    "jump rope",
    "laugh",
    "listen to music",
    "look back",
    "lunge",
    "make a phone call",
    "mop floor",
    "nod",
    "open umbrella",
    "pause and observe",
    "pick up",
    "play phone",
    "provide direction",
    "pull",
    "push",
    "push objects",
    "read book",
    "rub eyes",
    "rub hands",
    "run",
    "shake head",
    "shout",
    "sit down",
    "squat down",
    "stand",
    "stand up",
    "step back",
    "stretch",
    "support somebody",
    "sweep floor",
    "take photo",
    "talk",
    "tie shoes",
    "turn around",
    "walk",
    "wave",
    "wave to stop",
    "wear mask",
)

NUM_CLASSES = len(FOGACT_CLASSES)
FOGACT_SCENES = 10
FOGACT_PERSPECTIVES = ("front", "back", "left", "right")
FOG_INTENSITIES = ("light", "dense")


def action_prompt(label: int | str) -> str:
    name = FOGACT_CLASSES[label] if isinstance(label, int) else label
    return f"This is a video of '{name}'"
