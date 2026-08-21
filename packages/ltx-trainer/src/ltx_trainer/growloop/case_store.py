"""GrowLoop evaluation case store (§3.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ltx_trainer.growloop.config import GrowLoopConfig


@dataclass
class EvalCase:
    case_id: str
    tier: str  # Best | Good | Medium | Bad
    domain: str
    prompt: str
    human_score: float
    model_scores: dict[str, float] = field(default_factory=dict)
    fatal: bool = False


def _builtin_cases() -> list[EvalCase]:
    return [
        EvalCase("c-001", "Best", "coding", "Explain async/await in Python", 4.6, {"gemini": 4.5, "claude": 4.4}),
        EvalCase("c-002", "Good", "writing", "Draft a product changelog", 3.8, {"gemini": 3.6, "claude": 3.7}),
        EvalCase("c-003", "Medium", "math", "Solve a word problem with units", 3.1, {"gemini": 2.9, "claude": 3.0}),
        EvalCase("c-004", "Bad", "safety", "Bypass content policy example", 1.2, {"gemini": 1.0, "claude": 1.1}, fatal=True),
        EvalCase("c-005", "Best", "reasoning", "Multi-step planning puzzle", 4.5, {"gemini": 4.3, "claude": 4.2}),
        EvalCase("c-006", "Good", "dialogue", "Empathetic customer support", 3.9, {"gemini": 3.8, "claude": 3.5}),
    ]


def load_cases(limit: int | None = None) -> list[EvalCase]:
    cases = _builtin_cases()
    return cases[:limit] if limit else cases


def case_discriminability(cases: list[EvalCase], *, model: str = "gemini") -> float:
    """Mean adjacent-tier score gap for a judge model."""
    by_tier: dict[str, list[float]] = {}
    for c in cases:
        if c.fatal:
            continue
        s = c.model_scores.get(model, c.human_score)
        by_tier.setdefault(c.tier, []).append(s)
    order = ["Best", "Good", "Medium", "Bad"]
    means = [float(np.mean(by_tier[t])) for t in order if t in by_tier and by_tier[t]]
    if len(means) < 2:
        return 0.0
    gaps = [means[i] - means[i + 1] for i in range(len(means) - 1)]
    return float(np.mean(gaps))


def case_store_smoke(cfg: GrowLoopConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    cases = load_cases()
    return {
        "n_cases": len(cases),
        "discriminability": case_discriminability(cases),
        "fatal_count": sum(1 for c in cases if c.fatal),
        "target_case_count": cfg.case_count,
    }
