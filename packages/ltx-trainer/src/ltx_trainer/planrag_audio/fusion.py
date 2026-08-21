"""Nearest-midpoint temporal fusion toy (Appendix F, τ overlap)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeSegment:
    """A time-aligned row (start/end in seconds)."""

    start: float
    end: float
    payload: str


def _midpoint(seg: TimeSegment) -> float:
    return (seg.start + seg.end) / 2.0


def temporal_overlap(a: TimeSegment, b: TimeSegment, tau: float) -> bool:
    """True if intervals overlap when expanded by ±τ/2 on each side (overlap in expanded sense).

    Paper: overlap within tolerance window ±τ; we use standard interval overlap with margin τ.
    """
    return a.start - tau <= b.end and b.start - tau <= a.end


def fuse_nearest_midpoint(
    base: list[TimeSegment],
    target: list[TimeSegment],
    tau: float = 2.5,
) -> list[tuple[TimeSegment, TimeSegment | None]]:
    """For each base segment, pick at most one overlapping target with closest midpoint."""
    out: list[tuple[TimeSegment, TimeSegment | None]] = []
    for b in base:
        candidates = [t for t in target if temporal_overlap(b, t, tau)]
        if not candidates:
            out.append((b, None))
            continue
        mb = _midpoint(b)
        best = min(candidates, key=lambda t: abs(_midpoint(t) - mb))
        out.append((b, best))
    return out
