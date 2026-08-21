"""NV processor: style, discrete unit, duration parsers — Fig. 1."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.fgnv.annotation import NvSegment


@dataclass
class NvTokens:
    style: str
    discrete_count: int
    duration_units: float
    token_id: str


def style_parser(nv: NvSegment) -> str:
    return nv.style


def discrete_unit_parser(nv: NvSegment) -> int:
    """Count discrete syllables (e.g. 'ha ha ha' → 3)."""
    return len(nv.discrete_units)


def duration_parser(nv: NvSegment, base_seconds: float = 1.0, per_char_seconds: float = 0.2) -> float:
    """Continuous NV duration: base + 0.2 s per repeated last character (paper §3.2)."""
    if not nv.continuous_runs:
        return 0.0
    total = 0.0
    for _char, repeat in nv.continuous_runs:
        total += base_seconds + max(0, repeat - 1) * per_char_seconds
    return total


def nv_processor(nv: NvSegment) -> NvTokens:
    """Encode NV segment into structural tokens for Grad-TTS conditioning."""
    style = style_parser(nv)
    discrete = discrete_unit_parser(nv)
    if nv.discrete_units and not nv.continuous_runs:
        duration = float(discrete) * 0.15
    else:
        duration = duration_parser(nv)
    token_id = f"nv:{style}:d{discrete}:t{duration:.2f}"
    return NvTokens(style=style, discrete_count=discrete, duration_units=duration, token_id=token_id)


def encode_utterance(text: str) -> dict[str, object]:
    """Full pipeline for e.g. '<(crying) wuuuuu whep> why you do this to me.'"""
    from ltx_trainer.fgnv.annotation import split_verbal_nv

    nv, verbal = split_verbal_nv(text)
    if nv is None:
        return {"has_nv": False, "verbal": verbal, "tokens": None}
    tokens = nv_processor(nv)
    return {
        "has_nv": True,
        "nv_raw": nv.raw_tag,
        "verbal": verbal,
        "style": tokens.style,
        "discrete_count": tokens.discrete_count,
        "duration_units": tokens.duration_units,
        "token_id": tokens.token_id,
    }
