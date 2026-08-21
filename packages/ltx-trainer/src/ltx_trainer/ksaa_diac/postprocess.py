"""Positional diacritic insertion invariants — §3.4."""

from __future__ import annotations

# Toy Arabic letters without diacritics for smoke tests
_DIACRITIC_CHARS = "ًٌٍَُِّْ"


def strip_diacritics(text: str) -> str:
    return "".join(c for c in text if c not in _DIACRITIC_CHARS)


def insert_diacritics(
    undiacritized: str,
    diacritic_labels: list[str],
) -> str:
    """1:1 letter–diacritic correspondence (§3.4)."""
    if len(diacritic_labels) != len(undiacritized):
        raise ValueError("diacritic count must match letter positions")
    out = []
    for ch, d in zip(undiacritized, diacritic_labels, strict=True):
        out.append(ch)
        if d and d != "NONE":
            out.append(d)
    return "".join(out)


def verify_invariants(undiacritized: str, diacritized: str, labels: list[str]) -> dict[str, bool]:
    return {
        "strip_recover_input": strip_diacritics(diacritized) == undiacritized,
        "diacritic_count_matches": len(labels) == len(undiacritized),
        "all_positions_consumed": len(labels) == len(undiacritized),
    }
