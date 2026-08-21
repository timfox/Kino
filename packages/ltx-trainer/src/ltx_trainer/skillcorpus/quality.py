"""Three-facet quality scoring and Stage-5 gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.skillcorpus.config import HARD_GATE_FLAGS


@dataclass
class FacetScores:
    """Utility / robustness / safety subscores on [0, 1] (normalized from 0–10)."""

    utility: float
    robustness: float
    safety: float
    flags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "utility": round(self.utility, 4),
            "robustness": round(self.robustness, 4),
            "safety": round(self.safety, 4),
            "flags": list(self.flags),
        }


def content_quality(u: float, r: float, s: float) -> float:
    """content_q = 0.50 u + 0.35 r + 0.15 s, with marginal-safety attenuation."""
    q = 0.50 * u + 0.35 * r + 0.15 * s
    if 0.3 <= s <= 0.7:
        q *= 0.5 + 0.5 * (s - 0.3) / 0.4
    return max(0.0, min(1.0, q))


def composite_score(
    facets: FacetScores,
    *,
    prior_src: float = 0.685,
    has_scripts: bool = False,
    has_references: bool = False,
) -> float:
    """score = 0.85 content_q + 0.15 prior + structural bonus, clipped."""
    if any(f in HARD_GATE_FLAGS for f in facets.flags):
        return 0.0
    if facets.safety < 0.3:  # LLM safety subscore < 3/10
        return 0.0
    q = content_quality(facets.utility, facets.robustness, facets.safety)
    bonus = (0.05 if has_scripts else 0.0) + (0.02 if has_references else 0.0)
    return max(0.0, min(1.0, 0.85 * q + 0.15 * prior_src + bonus))


def hard_gate_blocks(flags: tuple[str, ...] | list[str]) -> bool:
    return any(f in HARD_GATE_FLAGS for f in flags)


def classify_skill(name: str, description: str, body_excerpt: str = "") -> str:
    """Heuristic 16-class stub (paper: LLM decision-tree classifier)."""
    text = f"{name} {description} {body_excerpt}".lower()
    # AI-ML only when primary deliverable is an AI system
    if any(k in text for k in ("persona agent", "agent persona", "train a model", "fine-tune llm")):
        return "AI-ML"
    rules = (
        ("Security", ("security", "vuln", "cve", "auth bypass", "pentest")),
        ("Testing", ("pytest", "unit test", "e2e test", "qa ")),
        ("DevOps-Infra", ("kubernetes", "terraform", "ci/cd", "docker compose", "devops")),
        ("Frontend-UI", ("react", "css", "frontend", "ui component")),
        ("Data", ("dataframe", "sql", "etl", "pandas", "analytics")),
        ("Writing", ("blog", "copywriting", "documentation prose", "essay")),
        ("Doc-Proc", ("pdf", "docx", "ocr", "document parse")),
        ("Multimedia", ("image", "video", "ffmpeg", "audio")),
        ("Workflow", ("orchestrat", "pipeline workflow", "automation workflow")),
        ("Meta", ("skill creator", "meta-skill", "harness")),
        ("Productivity", ("calendar", "todo", "notes")),
        ("Comms", ("email", "slack", "messaging")),
        ("Auth", ("oauth", "sso", "login flow")),
        ("Dev", ("git", "refactor", "code review", "python", "typescript", "api")),
    )
    for label, keys in rules:
        if any(k in text for k in keys):
            return label
    return "Other"
