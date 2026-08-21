"""Exact and fuzzy entity matching for VKB cell checks (Sec. 3.4, Table 3)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(a: str, b: str) -> bool:
    return normalize_text(a) == normalize_text(b)


def fuzzy_match(a: str, b: str, *, threshold: float = 0.85) -> bool:
    """Semantic similarity proxy via SequenceMatcher (paper: fuzzy_match)."""
    if not a or not b:
        return False
    na, nb = normalize_text(a), normalize_text(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def parse_time_hhmm(t: str) -> int | None:
    m = re.match(r"^(\d{1,2}):(\d{2})$", t.strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))
