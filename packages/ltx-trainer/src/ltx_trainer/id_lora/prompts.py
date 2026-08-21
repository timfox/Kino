"""Structured ID-LoRA prompt tags: [VISUAL], [SPEECH], [SOUNDS]."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ltx_trainer.id_lora.config import IdLoraConfig

_TAG_RE = re.compile(
    r"\[(VISUAL|SPEECH|SOUNDS)\]:\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedIdLoraPrompt:
    visual: str
    speech: str
    sounds: str
    raw: str

    @property
    def is_structured(self) -> bool:
        return bool(self.visual.strip() or self.speech.strip() or self.sounds.strip())

    @property
    def identity_ready(self) -> bool:
        return bool(self.visual.strip() and self.speech.strip())


def parse_id_lora_prompt(text: str, _cfg: IdLoraConfig | None = None) -> ParsedIdLoraPrompt:
    """Parse tagged sections; missing tags yield empty strings."""
    raw = text.strip()
    if not raw:
        return ParsedIdLoraPrompt(visual="", speech="", sounds="", raw=raw)

    parts: dict[str, str] = {"visual": "", "speech": "", "sounds": ""}
    matches = list(_TAG_RE.finditer(raw))
    if not matches:
        return ParsedIdLoraPrompt(visual=raw, speech="", sounds="", raw=raw)

    for i, m in enumerate(matches):
        key = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        parts[key] = raw[start:end].strip()

    return ParsedIdLoraPrompt(
        visual=parts["visual"],
        speech=parts["speech"],
        sounds=parts["sounds"],
        raw=raw,
    )


def build_structured_prompt(
    visual: str,
    speech: str,
    sounds: str = "",
) -> str:
    """Build canonical ID-LoRA prompt string."""
    blocks = [f"[VISUAL]: {visual.strip()}"]
    if speech.strip():
        blocks.append(f"[SPEECH]: {speech.strip()}")
    if sounds.strip():
        blocks.append(f"[SOUNDS]: {sounds.strip()}")
    return "\n".join(blocks)
