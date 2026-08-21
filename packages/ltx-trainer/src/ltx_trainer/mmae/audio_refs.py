"""Parse `<audio label[start:end]>` references in rubric questions (Appendix C)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_AUDIO_REF_RE = re.compile(
    r"<audio\s+(?P<label>[^>\[]+)"
    r"(?:\[(?P<start>[^\]:]*)(?::(?P<end>[^\]]*))?\])?"
    r"\s*>"
)


@dataclass(frozen=True)
class AudioRef:
    label: str
    start_sec: float | None = None
    end_sec: float | None = None

    @property
    def is_slice(self) -> bool:
        return self.start_sec is not None or self.end_sec is not None


def _parse_time(token: str | None) -> float | None:
    if token is None or token == "":
        return None
    token = token.strip()
    if token.endswith("s"):
        token = token[:-1].strip()
    if not token:
        return None
    if token.startswith("-"):
        return float(token)
    return float(token)


def parse_audio_ref(text: str) -> AudioRef | None:
    match = _AUDIO_REF_RE.search(text.strip())
    if not match:
        return None
    label = match.group("label").strip()
    return AudioRef(
        label=label,
        start_sec=_parse_time(match.group("start")),
        end_sec=_parse_time(match.group("end")),
    )


def parse_audio_refs(text: str) -> list[AudioRef]:
    refs: list[AudioRef] = []
    for match in _AUDIO_REF_RE.finditer(text):
        label = match.group("label").strip()
        refs.append(
            AudioRef(
                label=label,
                start_sec=_parse_time(match.group("start")),
                end_sec=_parse_time(match.group("end")),
            )
        )
    return refs
