"""Seed-event long-tail and temporal filters (Appendix A, Table 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SourceKind = Literal["gdelt", "tmdb", "rawg", "cve", "sports", "usgs"]


@dataclass
class SeedEvent:
    source: SourceKind
    event_id: str
    published_days_ago: int
    metadata: dict[str, Any]


def temporal_pass(event: SeedEvent, *, max_age_days: int = 90) -> bool:
    return 0 <= event.published_days_ago <= max_age_days


def long_tail_score_tmdb(meta: dict[str, Any]) -> float:
    score = 0.0
    pop = float(meta.get("popularity", 10))
    votes = int(meta.get("vote_count", 100))
    revenue = float(meta.get("revenue", 1))
    if pop <= 1:
        score += 1.0
    elif pop <= 5:
        score += 0.5
    if votes <= 20:
        score += 1.0
    elif votes <= 50:
        score += 0.5
    if revenue <= 0:
        score += 1.0
    if meta.get("non_english"):
        score += 0.5
    return score


def long_tail_score_rawg(meta: dict[str, Any]) -> float:
    score = 0.0
    ratings = int(meta.get("ratings_count", 50))
    added = int(meta.get("added", 10))
    metacritic = meta.get("metacritic")
    if ratings <= 10:
        score += 1.0
    if added <= 1:
        score += 1.0
    if metacritic is None:
        score += 1.0
    elif float(metacritic) >= 85:
        score += 0.0
    return score


def long_tail_score_cve(meta: dict[str, Any]) -> float:
    score = 0.0
    cvss = float(meta.get("cvss", 5))
    if cvss >= 9:
        score += 1.0
    elif cvss >= 7:
        score += 0.5
    if meta.get("single_product"):
        score += 1.0
    if meta.get("exploit_available"):
        score += 0.5
    if int(meta.get("published_days_ago", 90)) <= 30:
        score += 1.0
    return score


def long_tail_score_sports(meta: dict[str, Any]) -> float:
    score = 0.0
    name = str(meta.get("event_name", "")).lower()
    for kw in ("final", "championship", "cup"):
        if kw in name:
            score += 0.5
    if int(meta.get("attendance", 0)) > 50_000:
        score += 0.5
    if meta.get("sport") != "football":
        score += 1.5
    return score


def long_tail_score_usgs(meta: dict[str, Any]) -> float:
    score = 0.0
    mag = float(meta.get("magnitude", 4))
    if mag >= 7:
        score += 1.0
    elif mag >= 5:
        score += 0.5
    elif mag <= 4:
        score += 0.25
    sig = int(meta.get("significance", 0))
    if sig >= 600:
        score += 1.0
    depth = float(meta.get("depth_km", 10))
    if depth < 10 or depth > 500:
        score += 0.5
    return score


def long_tail_score_gdelt(meta: dict[str, Any]) -> float:
    """LLM heat 2.0–4.0 band per paper."""
    heat = float(meta.get("llm_heat", 3.0))
    length = int(meta.get("article_chars", 200))
    if length < 150:
        return 0.0
    if 2.0 <= heat <= 4.0:
        return 2.5
    return 0.0


_THRESHOLDS: dict[SourceKind, float] = {
    "gdelt": 2.0,
    "tmdb": 2.5,
    "rawg": 2.5,
    "cve": 2.0,
    "sports": 1.5,
    "usgs": 1.5,
}


def long_tail_score(event: SeedEvent) -> float:
    m = event.metadata
    if event.source == "tmdb":
        return long_tail_score_tmdb(m)
    if event.source == "rawg":
        return long_tail_score_rawg(m)
    if event.source == "cve":
        return long_tail_score_cve(m)
    if event.source == "sports":
        return long_tail_score_sports(m)
    if event.source == "usgs":
        return long_tail_score_usgs(m)
    return long_tail_score_gdelt(m)


def filter_seed_pipeline(event: SeedEvent, *, max_age_days: int = 90) -> dict[str, Any]:
    """Run temporal + long-tail + answer-stability gates."""
    temporal_ok = temporal_pass(event, max_age_days=max_age_days)
    score = long_tail_score(event)
    threshold = _THRESHOLDS.get(event.source, 2.0)
    long_tail_ok = score >= threshold
    stable = bool(event.metadata.get("answer_stable", True))
    return {
        "event_id": event.event_id,
        "source": event.source,
        "temporal_ok": temporal_ok,
        "long_tail_score": round(score, 2),
        "long_tail_ok": long_tail_ok,
        "answer_stable": stable,
        "passes": temporal_ok and long_tail_ok and stable,
    }
