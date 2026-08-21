"""Automatic generation metrics (Appendix D)."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_PITCH_RE = re.compile(r"[\^_=]?[A-Ga-g][,']*")
_SCALE_C = frozenset({"C", "D", "E", "F", "G", "A", "B", "_E", "_A", "_B"})


def extract_pitches(abc: str) -> list[str]:
    pitches = []
    for m in _PITCH_RE.findall(abc):
        p = m.replace(",", "").replace("'", "").upper()
        if p.startswith("^") or p.startswith("="):
            return []  # non-adherent sharp/natural marks whole score
        if p.startswith("_"):
            pitches.append("_" + p[1:])
        elif p:
            pitches.append(p[0])
    return pitches


def pitch_histogram(pitches: list[str]) -> dict[str, float]:
    if not pitches:
        return {}
    counts = Counter(pitches)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def kl_divergence(p: dict[str, float], q: dict[str, float], *, eps: float = 1e-10) -> float:
    keys = set(p) | set(q)
    dkl = 0.0
    for k in keys:
        pv = p.get(k, 0.0) + eps
        qv = q.get(k, 0.0) + eps
        dkl += pv * math.log(pv / qv)
    return float(dkl)


def pitch_entropy(pitches: list[str]) -> float:
    hist = pitch_histogram(pitches)
    if not hist:
        return 0.0
    return float(-sum(v * math.log2(v) for v in hist.values() if v > 0))


def repetition_rate(abc_scores: list[str]) -> float:
    if not abc_scores:
        return 0.0
    hits = sum(1 for s in abc_scores if "|:" in s)
    return hits / len(abc_scores)


def scale_adherence(abc: str) -> bool:
    pitches = extract_pitches(abc)
    if not pitches:
        return False
    return all(p in _SCALE_C for p in pitches)


def scale_adherence_rate(abc_scores: list[str]) -> float:
    if not abc_scores:
        return 0.0
    return sum(scale_adherence(s) for s in abc_scores) / len(abc_scores)


def abc_syntax_valid(abc: str) -> bool:
    """Lightweight ABC validity stub (header + note tokens)."""
    return "K:" in abc and bool(_PITCH_RE.search(abc))


def abc_syntax_rate(abc_scores: list[str]) -> float:
    if not abc_scores:
        return 0.0
    return sum(abc_syntax_valid(s) for s in abc_scores) / len(abc_scores)


def pearson_r(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
