"""AdaptiveCacheBoundary — STATIC/QUASI/DYNAMIC prefix (arXiv:2607.15516 §5.3)."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ltx_trainer.capc.config import CapcConfig
from ltx_trainer.capc.cost_model import rho_empirical


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
_AMOUNT_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d+\.\d{2}\s*(?:USD|EUR|GBP)\b", re.I)
_NUM_RE = re.compile(r"\b\d{4,}\b")


class SegmentClass(str, Enum):
    STATIC = "STATIC"
    QUASI = "QUASI"
    DYNAMIC = "DYNAMIC"


@dataclass(frozen=True)
class Segment:
    text: str
    cls: SegmentClass
    mutation_rate: float
    position: int


def normalize(text: str) -> str:
    """Regex normalizer: dates → [DATE], amounts → [AMOUNT], large nums → [NUM]."""
    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = _DATE_RE.sub("[DATE]", s)
    s = _AMOUNT_RE.sub("[AMOUNT]", s)
    s = _NUM_RE.sub("[NUM]", s)
    return s.strip()


def sentences(doc: str) -> list[str]:
    raw = (doc or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(raw) if p and p.strip()]
    return parts or [raw]


def _fingerprint(sent: str) -> str:
    return hashlib.md5(normalize(sent).encode("utf-8")).hexdigest()


class AdaptiveCacheBoundary:
    """Classify sentence positions by mutation rate across observed versions."""

    def __init__(
        self,
        *,
        eps_static: float | None = None,
        eps_quasi: float | None = None,
        min_calls: int | None = None,
        cfg: CapcConfig | None = None,
    ) -> None:
        c = cfg or CapcConfig()
        self.eps_static = float(c.eps_static if eps_static is None else eps_static)
        self.eps_quasi = float(c.eps_quasi if eps_quasi is None else eps_quasi)
        self.min_calls = int(c.min_calls if min_calls is None else min_calls)
        self._position_hashes: dict[int, list[str]] = defaultdict(list)
        self._seg_text: dict[str, str] = {}
        self._call_count = 0

    def observe(self, doc: str) -> None:
        """Feed one observed document version."""
        sents = sentences(doc)
        self._call_count += 1
        for pos, sent in enumerate(sents):
            h = _fingerprint(sent)
            self._position_hashes[pos].append(h)
            self._seg_text[h] = sent

    def classify(self) -> list[Segment]:
        """Classify each position by mutation rate μ = 1 − max_count/K."""
        out: list[Segment] = []
        for pos in sorted(self._position_hashes):
            hashes = self._position_hashes[pos]
            if len(hashes) < self.min_calls:
                text = self._seg_text.get(hashes[-1], "") if hashes else ""
                out.append(
                    Segment(text=text, cls=SegmentClass.DYNAMIC, mutation_rate=1.0, position=pos)
                )
                continue
            counts = Counter(hashes)
            most_common_h, most_common_count = counts.most_common(1)[0]
            mu = 1.0 - most_common_count / len(hashes)
            if mu <= self.eps_static:
                cls = SegmentClass.STATIC
            elif mu <= self.eps_quasi:
                cls = SegmentClass.QUASI
            else:
                cls = SegmentClass.DYNAMIC
            out.append(
                Segment(
                    text=self._seg_text.get(most_common_h, ""),
                    cls=cls,
                    mutation_rate=float(mu),
                    position=pos,
                )
            )
        return out

    def build_cache_prefix(self) -> str:
        """Maximal contiguous STATIC/QUASI prefix."""
        parts: list[str] = []
        for seg in self.classify():
            if seg.cls == SegmentClass.DYNAMIC or not seg.text:
                break
            parts.append(seg.text)
        return " ".join(parts)

    def composition(self) -> dict[str, Any]:
        segs = self.classify()
        n = max(1, len(segs))
        counts = Counter(s.cls.value for s in segs)
        return {
            "n_positions": len(segs),
            "static_frac": counts.get("STATIC", 0) / n,
            "quasi_frac": counts.get("QUASI", 0) / n,
            "dynamic_frac": counts.get("DYNAMIC", 0) / n,
            "prefix_chars": len(self.build_cache_prefix()),
            "versions": self._call_count,
        }


def synthetic_version_drift(
    *,
    base_tokens: int = 6000,
    k: int = 25,
    mutation_rate: float = 0.10,
    seed: int = 0,
) -> list[str]:
    """Build K document versions with controlled sentence mutation rate.

    ``mutation_rate`` sets the fraction of sentence positions that are DYNAMIC
    (always mutate every version). Remaining positions are STATIC or lightly
    QUASI (occasional date/amount edits).
    """
    state = seed & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / 0x100000000

    n_sents = max(8, base_tokens // 40)
    # mutation_rate ≈ DYNAMIC share; rest split STATIC / QUASI.
    dyn_frac = float(max(0.05, min(0.80, mutation_rate)))
    dyn_n = max(1, int(round(n_sents * dyn_frac)))
    rest = max(2, n_sents - dyn_n)
    static_n = max(1, int(round(rest * 0.6)))
    quasi_n = max(1, rest - static_n)

    static = [f"Static policy clause {i} remains binding." for i in range(static_n)]
    versions: list[str] = []
    for v in range(k):
        quasi: list[str] = []
        for i in range(quasi_n):
            # Soft drift: occasional date/amount change (μ in QUASI band).
            if rnd() < 0.15:
                quasi.append(f"Quasi schedule item {i} on 2026-0{(v % 9) + 1}-15 costs ${100 + v}.")
            else:
                quasi.append(f"Quasi schedule item {i} on 2026-01-15 costs $100.")
        dyn: list[str] = []
        for i in range(dyn_n):
            # Always mutate — μ → 1.0 → DYNAMIC.
            dyn.append(f"Dynamic note {i} revision {v} payload {int(rnd() * 1e6)}.")
        versions.append(" ".join(static + quasi + dyn))
    return versions


def adaptive_vs_naive_savings(
    versions: list[str],
    *,
    n_queries: int = 10,
    cfg: CapcConfig | None = None,
) -> dict[str, Any]:
    """Compare adaptive prefix caching vs naïve full-doc cache-all (Table 5/6).

    Naïve cache-all on an *evolving* document pays the cache-write rate on the
    full doc every call (prefix-strict miss). Adaptive caches only the
    STATIC/QUASI contiguous prefix and sends the dynamic remainder uncached.
    """
    c = cfg or CapcConfig()
    boundary = AdaptiveCacheBoundary(cfg=c)
    for doc in versions:
        boundary.observe(doc)
    prefix = boundary.build_cache_prefix()
    latest = versions[-1] if versions else ""
    prefix_tok = max(1, len(prefix) // 4) if prefix else 1
    full_tok = max(1, len(latest) // 4)
    dyn_tok = max(0, full_tok - prefix_tok)
    dq = 50
    o = 50
    n = max(1, int(n_queries))

    pin = c.pin / 1e6
    cw = c.cw / 1e6
    cr = c.cr / 1e6
    pout = c.pout / 1e6

    # Naïve: mutating full doc → cache miss every call → write tax on |D|.
    naive_cost = full_tok * cw + dq * pin + o * pout

    # Adaptive: stable prefix hits cache; dynamic remainder at pin.
    rho_p = rho_empirical(n, prefix_tok, c)
    adaptive_cost = (
        (1.0 - rho_p) * prefix_tok * cw
        + rho_p * prefix_tok * cr
        + (dyn_tok + dq) * pin
        + o * pout
    )
    savings = (naive_cost - adaptive_cost) / max(1e-12, naive_cost)
    comp = boundary.composition()
    return {
        "prefix_tokens_est": prefix_tok,
        "full_tokens_est": full_tok,
        "naive_cost": naive_cost,
        "adaptive_cost": adaptive_cost,
        "savings_frac": float(savings),
        "positive_savings": savings > 0,
        "composition": comp,
        "prefix_preview": prefix[:200],
    }
