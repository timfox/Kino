"""Fine-grained NV transcription tags — §3.2, Table 2."""

from __future__ import annotations

import re
from dataclasses import dataclass

NV_TAG_PATTERN = re.compile(
    r"<\((?P<style>[^)]+)\)\s*(?P<body>[^>]*)>",
    re.IGNORECASE,
)

NV_CATEGORIES = (
    "cheering",
    "yelling",
    "laughter-open",
    "laughter-closed",
    "crying",
    "screaming",
)


@dataclass
class NvSegment:
    style: str
    body: str
    discrete_units: list[str]
    continuous_runs: list[tuple[str, int]]  # (token, repeat count of last char)

    @property
    def raw_tag(self) -> str:
        return f"<({self.style}) {self.body}>".strip()


def parse_nv_segment(tag_text: str) -> NvSegment | None:
    """Parse '<(crying) wuuuuu whep>' style tags."""
    m = NV_TAG_PATTERN.search(tag_text.strip())
    if not m:
        return None
    style = m.group("style").strip().lower()
    body = m.group("body").strip().lower()
    tokens = body.split() if body else []
    discrete: list[str] = []
    continuous: list[tuple[str, int]] = []
    for tok in tokens:
        tok = tok.lower()
        if len(tok) >= 2 and len(set(tok)) == 1 and tok[0].isalpha():
            continuous.append((tok[0], len(tok)))
        elif len(tok) >= 4 and len(set(tok[1:])) == 1 and tok[1].isalpha():
            # e.g. wuuuuu → prolong trailing character (paper §3.2)
            continuous.append((tok[-1], len(tok) - 1))
        else:
            discrete.append(tok)
    return NvSegment(style=style, body=body, discrete_units=discrete, continuous_runs=continuous)


def split_verbal_nv(text: str) -> tuple[NvSegment | None, str]:
    """Split leading NV tag from verbal content."""
    text = text.strip()
    m = NV_TAG_PATTERN.match(text)
    if not m:
        return None, text
    nv = parse_nv_segment(m.group(0))
    verbal = text[m.end() :].strip()
    return nv, verbal


def to_coarse_tag(style: str) -> str:
    """Map fine style to coarse <crying> style tag."""
    return f"<{style.split('-')[0]}>"
