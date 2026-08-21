"""Curation funnel and retrieval-and-selection stack stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.skillcorpus.quality import FacetScores, classify_skill, composite_score, hard_gate_blocks


# Approximate funnel stages from paper narrative
FUNNEL_STAGES: list[tuple[str, int]] = [
    ("raw_crawl", 821000),
    ("post_parse_dedup_input", 283844),
    ("post_dedup", 101111),
    ("active_release", 96401),
]


@dataclass
class SkillRecord:
    skill_id: str
    name: str
    description: str
    body: str
    taxonomy: str
    score: float
    facets: FacetScores
    active: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description[:120],
            "body": self.body[:4000],
            "taxonomy": self.taxonomy,
            "score": round(self.score, 4),
            "facets": self.facets.as_dict(),
            "active": self.active,
        }


def curation_funnel_summary() -> dict[str, Any]:
    stages = [{"stage": s, "n": n} for s, n in FUNNEL_STAGES]
    raw = FUNNEL_STAGES[0][1]
    active = FUNNEL_STAGES[-1][1]
    return {
        "stages": stages,
        "reduction_pct": round(100.0 * (1.0 - active / raw), 2),
        "n_active": active,
        "n_raw": raw,
        "licence": "100% OSI-permissive",
        "taxonomy_classes": 16,
        "quality_facets": ("utility", "robustness", "safety"),
    }


def demo_skills() -> list[SkillRecord]:
    """Tiny in-memory corpus for CPU demos."""
    specs = [
        (
            "mfg-codebook",
            "manufacturing-failure-reason-codebook-normalization",
            "Normalise defect reasons into a station-scoped codebook with calibrated confidence.",
            "If entry.stations is not None, predicted code valid only when station matches. "
            "Multi-evidence scoring: fuzzy overlap, station compatibility, fail_code alignment.",
            ("",),
            0.9,
            0.85,
            0.95,
        ),
        (
            "fuzzy-match",
            "fuzzy-match",
            "Fuzzy string matching utility for codebook alignment.",
            "Use sequence ratio and token set ratio; return best candidate above threshold.",
            ("",),
            0.8,
            0.8,
            0.9,
        ),
        (
            "pptx-edit",
            "powerpoint-table-edit",
            "Edit Excel tables and PowerPoint slides via openpyxl/python-pptx.",
            "load_workbook('file.xlsx'); Presentation('file.pptx') — flat-file recipes only.",
            ("",),
            0.85,
            0.75,
            0.9,
        ),
        (
            "malware-skill",
            "credential-exfil-helper",
            "Helpful ops automation.",
            "Uses os.system with user input; may leak secrets.",
            ("cmd_injection", "unsafe_exec"),
            0.7,
            0.6,
            0.1,
        ),
    ]
    out: list[SkillRecord] = []
    for sid, name, desc, body, flags, u, r, s in specs:
        facets = FacetScores(u, r, s, tuple(flags))
        tax = classify_skill(name, desc, body)
        score = composite_score(facets, has_scripts=True)
        active = not hard_gate_blocks(flags) and s >= 0.3
        out.append(
            SkillRecord(
                skill_id=sid,
                name=name,
                description=desc,
                body=body,
                taxonomy=tax,
                score=score,
                facets=facets,
                active=active,
            )
        )
    return out


@dataclass
class RetrievalHit:
    skill: SkillRecord
    rerank_score: float

    def as_dict(self) -> dict[str, Any]:
        return {"skill": self.skill.as_dict(), "rerank_score": round(self.rerank_score, 4)}


def _overlap(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q:
        return 0.0
    return len(q & t) / len(q)


def retrieve_and_select(query: str, corpus: list[SkillRecord] | None = None, top_k: int = 2) -> dict[str, Any]:
    """Recall → rerank → LLM-selector stub (0–2 skills)."""
    pool = [s for s in (corpus or demo_skills()) if s.active]
    scored: list[RetrievalHit] = []
    for s in pool:
        field = f"{s.name} {s.description} {s.body}"[:3000]
        score = 0.6 * _overlap(query, field) + 0.4 * s.score
        scored.append(RetrievalHit(s, score))
    scored.sort(key=lambda h: h.rerank_score, reverse=True)
    # Selector: keep 0–2 with score above soft threshold
    selected = [h for h in scored[: max(top_k, 3)] if h.rerank_score >= 0.25][:top_k]
    return {
        "query": query[:200],
        "n_candidates": len(pool),
        "reranked": [h.as_dict() for h in scored[:5]],
        "selected": [h.as_dict() for h in selected],
        "n_selected": len(selected),
        "top_rerank": scored[0].rerank_score if scored else 0.0,
    }


def coverage_bin(top_rerank: float) -> str:
    if top_rerank < 0.45:
        return "low"
    if top_rerank < 0.75:
        return "medium"
    return "high"


# Paper Fig. 5 mean Δ by bin
COVERAGE_BIN_DELTA: dict[str, float] = {
    "low": 2.2,
    "medium": 6.2,
    "high": 25.1,
}
