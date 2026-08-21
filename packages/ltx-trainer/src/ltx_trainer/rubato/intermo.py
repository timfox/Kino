"""InterMo: intervals-and-moments text notation (Sec. 2, Fig. 1)."""

from __future__ import annotations

import re
from typing import Final

# Staff markers (Fig. 1).
STAFF_RIGHT: Final[str] = "PR:"
STAFF_LEFT: Final[str] = "PL:"

# Structural barline pattern: |meter k-accidentals e.g. |3/4k-4
BARLINE_PATTERN = re.compile(r"\|(\d+)/(\d+)k(-?\d+)")


def parse_barline(token: str) -> tuple[int, int, int] | None:
    """Parse a barline token like ``|3/4k-4`` → (3, 4, -4)."""
    m = BARLINE_PATTERN.match(token.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def metric_interval_tokens(text: str) -> list[str]:
    """Extract simple rational metric tokens ``1/8``, ``1/4``, etc. from a fragment."""
    return re.findall(r"\b\d+/\d+\b", text)


def sum_metric_intervals(intervals: list[str]) -> float | None:
    """Sum ``num/den`` fractions; returns None if any token invalid."""
    total = 0.0
    for t in intervals:
        if "/" not in t:
            return None
        n, d = t.split("/", 1)
        di = int(d)
        if di == 0:
            return None
        total += int(n) / di
    return total


def validate_measure_metric_sum(barline: str, intervals_in_measure: list[str], *, eps: float = 1e-6) -> bool:
    r"""Check intervals between barlines sum to declared meter (Sec. 2, local metric arithmetic)."""
    parsed = parse_barline(barline)
    if not parsed:
        return False
    num, den, _ = parsed
    expected = num / den
    s = sum_metric_intervals(intervals_in_measure)
    if s is None:
        return False
    return abs(s - expected) < eps


def tast_first_bar_example() -> str:
    """First-bar TAST excerpt from Fig. 2 (truncated after a few tokens for doc/tests)."""
    return (
        "|3/4k-4 PR: C5 <|0.20|> 1/4 PL: A-3 C4 F4 <|1.40|> 1/8 PR: c5 D-5 <|1.80|> "
        "1/8 PL: a-3 c4 f4 A-3 C4 F4 PR: d-5 D-5 <|2.10|>"
    )


def pitch_case_is_onset(token: str) -> bool:
    """Heuristic: note name starting with A–G uppercase → onset; lowercase → offset."""
    if len(token) < 2:
        return False
    return token[0].isupper() and token[0] in "ABCDEFG"
