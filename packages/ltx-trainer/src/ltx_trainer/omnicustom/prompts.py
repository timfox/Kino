"""Speech span parsing for OmniCustom prompts (<S>…<E>)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ltx_trainer.omnicustom.config import OmniCustomConfig


@dataclass(frozen=True)
class ParsedPrompt:
    scene_text: str
    speech_text: str | None
    has_speech: bool
    raw: str


_SPEECH_RE = re.compile(r"<S>(.*?)<E>", re.DOTALL | re.IGNORECASE)


def extract_speech(text: str, cfg: OmniCustomConfig | None = None) -> ParsedPrompt:
    """Split scene description from marked speech content."""
    c = cfg or OmniCustomConfig()
    raw = text.strip()
    m = _SPEECH_RE.search(raw)
    if not m:
        return ParsedPrompt(scene_text=raw, speech_text=None, has_speech=False, raw=raw)
    speech = m.group(1).strip()
    scene = (raw[: m.start()] + raw[m.end() :]).strip()
    scene = re.sub(r"\s+", " ", scene)
    return ParsedPrompt(scene_text=scene, speech_text=speech, has_speech=bool(speech), raw=raw)


def strip_speech_tags(text: str) -> str:
    """Remove <S>/<E> markers for baselines that do not support inline speech."""
    return _SPEECH_RE.sub(lambda m: m.group(1), text)
