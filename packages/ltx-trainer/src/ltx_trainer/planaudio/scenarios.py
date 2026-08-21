"""Free-form prompt scenarios (Table 1) and classification helpers."""

from __future__ import annotations

import re
from typing import Any

SCENARIO_SOUND = "sound"
SCENARIO_SPEECH = "speech"
SCENARIO_COMPOSITE = "composite"

TABLE1_EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "scenario": SCENARIO_SOUND,
        "free_form": "An acoustic guitar plays, then a man sings.",
        "rewritten": "[Sound]: An acoustic guitar plays, then a man sings.\n[Speech]: singing",
    },
    {
        "scenario": SCENARIO_SPEECH,
        "free_form": '"Jesus Christ!", a man says, without background sound.',
        "rewritten": "[Sound]: clean speech for an audiobook.\n[Speech]: Jesus Christ!",
    },
    {
        "scenario": SCENARIO_COMPOSITE,
        "free_form": 'An acoustic guitar plays, then a man sings: "Jesus Christ!"',
        "rewritten": "[Sound]: An acoustic guitar plays, then a man sings.\n[Speech]: Jesus Christ!",
    },
)

_QUOTED_SPEECH = re.compile(r'"([^"]{2,})"|\'([^\']{2,})\'|says[:\s]+([^.!?]+)', re.I)


def classify_free_form_prompt(prompt: str) -> str:
    """Heuristic scenario tag for LTX caption / manifest routing (CPU stub)."""
    text = prompt.strip()
    if not text:
        return SCENARIO_SOUND
    has_quote = bool(_QUOTED_SPEECH.search(text))
    clean_speech = "without background" in text.lower() or "clean speech" in text.lower()
    sound_cues = (
        "applause",
        "music",
        "guitar",
        "drum",
        "rain",
        "thunder",
        "door",
        "footstep",
        "laugh",
        "crowd",
        "ambient",
        "background",
    )
    has_sound = any(c in text.lower() for c in sound_cues)

    if clean_speech and has_quote:
        return SCENARIO_SPEECH
    if has_quote and (has_sound or "then" in text.lower() or "followed" in text.lower()):
        return SCENARIO_COMPOSITE
    if has_quote:
        return SCENARIO_SPEECH
    return SCENARIO_SOUND


def scenario_card() -> dict[str, Any]:
    return {
        "sound": "Non-linguistic / environmental; speech may appear without exact transcript.",
        "speech": "Linguistic constraints with precise transcriptions.",
        "composite": "Joint speech content and sound events with interaction.",
        "examples": list(TABLE1_EXAMPLES),
    }
