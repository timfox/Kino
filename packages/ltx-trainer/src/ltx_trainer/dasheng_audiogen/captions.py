"""Structured multi-view captions (Dasheng AudioGen §3.1)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CAPTION_VIEWS: tuple[str, ...] = (
    "<|caption|>",
    "<|speech|>",
    "<|asr|>",
    "<|sfx|>",
    "<|music|>",
    "<|env|>",
)

REQUIRED_VIEW = "<|caption|>"
_VIEW_RE = re.compile(r"<\|(caption|speech|asr|sfx|music|env)\|>")


@dataclass
class StructuredCaption:
    caption: str
    speech: str | None = None
    asr: str | None = None
    sfx: str | None = None
    music: str | None = None
    env: str | None = None

    def to_sequence(self) -> list[tuple[str, str]]:
        """View segments [s_k, y_k] for T5 encoding."""
        out: list[tuple[str, str]] = [(REQUIRED_VIEW, self.caption)]
        for token, val in (
            ("<|speech|>", self.speech),
            ("<|asr|>", self.asr),
            ("<|sfx|>", self.sfx),
            ("<|music|>", self.music),
            ("<|env|>", self.env),
        ):
            if val:
                out.append((token, val))
        return out

    def to_flat_text(self) -> str:
        return self.caption

    def serialize(self) -> str:
        """Round-trip special-token string."""
        parts = [f"{REQUIRED_VIEW} {self.caption}"]
        for token, val in (
            ("<|speech|>", self.speech),
            ("<|asr|>", self.asr),
            ("<|sfx|>", self.sfx),
            ("<|music|>", self.music),
            ("<|env|>", self.env),
        ):
            if val:
                parts.append(f"{token} {val}")
        return " ".join(parts)


def parse_structured_caption(text: str) -> StructuredCaption:
    """Parse inline special-token delimited caption string."""
    matches = list(_VIEW_RE.finditer(text))
    if not matches:
        return StructuredCaption(caption=text.strip())

    parts: dict[str, str] = {}
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts[key] = text[start:end].strip()

    return StructuredCaption(
        caption=parts.get("caption", text.strip()),
        speech=parts.get("speech"),
        asr=parts.get("asr"),
        sfx=parts.get("sfx"),
        music=parts.get("music"),
        env=parts.get("env"),
    )


def caption_smoke() -> dict[str, Any]:
    sample = StructuredCaption(
        caption="Female voice speaking over synth-driven pop music and subtle ambient sounds.",
        speech="Female voice explaining a hair preparation process in German.",
        asr="lockern und einige Strähnen rausziehen.",
        sfx="Ambiguous environmental noises suggesting manual activity.",
        music="Electronically processed guitar patterns with pop music structure.",
        env="Indoor recording with mild electrical interference.",
    )
    seq = sample.to_sequence()
    serialized = sample.serialize()
    roundtrip = parse_structured_caption(serialized)
    return {
        "n_views": len(CAPTION_VIEWS),
        "n_active_views": len(seq),
        "has_required_caption": seq[0][0] == REQUIRED_VIEW,
        "flat_shorter": len(sample.to_flat_text()) < sum(len(y) for _, y in seq),
        "roundtrip_asr": roundtrip.asr == sample.asr,
    }
