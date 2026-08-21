"""Physical-faithfulness judge scoring stub (Sec. 3.3)."""

from __future__ import annotations

from typing import Any


PHYSICS_EVENT_CLASSES: tuple[str, ...] = (
    "A_collision_rebound",
    "B_destruction_deformation",
    "C_fluids_liquids",
    "D_shadow_reflection",
    "E_chain_multi_stage",
    "F_rolling_sliding",
    "G_throwing_ballistic",
)


def overall_physics_score(
    sa: float,
    ptv: float,
    persistence: float,
    solid_body: float,
    fluid: float,
    optical: float,
) -> float:
    """Table 4 Overall = 0.5 * mean(SA, PTV, Persist.) + 0.5 * mean(domain laws)."""
    general = (sa + ptv + persistence) / 3.0
    physics = (solid_body + fluid + optical) / 3.0
    return round(0.5 * general + 0.5 * physics, 2)


def judge_dimensions() -> dict[str, list[str]]:
    return {
        "general": ["semantic_alignment", "physical_temporal_validity", "persistence"],
        "physics_domains": ["solid_body", "fluid", "optical"],
        "per_law": list(PHYSICS_EVENT_CLASSES),
    }


def round4_trainset_quotas() -> list[dict[str, Any]]:
    """Table 2 — DPO trainset class balance (1,000 pairs)."""
    return [
        {"class": "A (collision/rebound)", "pairs": 513},
        {"class": "B (destruction/deformation)", "pairs": 93},
        {"class": "C (fluids/liquids)", "pairs": 168},
        {"class": "D (shadow/reflection)", "pairs": 68},
        {"class": "E (chain/multi-stage)", "pairs": 13},
        {"class": "F (rolling/sliding)", "pairs": 75},
        {"class": "G (throwing/ballistic)", "pairs": 55},
        {"class": "unclassified", "pairs": 15},
    ]


def physics_filter_decision(
    *,
    physics_proxy: float | None = None,
    overall_score: float | None = None,
    min_proxy: float = 3.25,
    min_overall: float = 62.0,
) -> dict[str, Any]:
    """Post-train filter decision for generated clips.

    ``physics_proxy`` is the 1–5 fold-sidecar scale. ``overall_score`` is the judge
    scale returned by :func:`overall_physics_score` when a richer evaluator is used.
    A clip passes if every provided score clears its threshold.
    """
    checks: dict[str, bool] = {}
    if physics_proxy is not None:
        checks["physics_proxy"] = float(physics_proxy) >= min_proxy
    if overall_score is not None:
        checks["overall_score"] = float(overall_score) >= min_overall
    passed = bool(checks) and all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "thresholds": {"min_proxy": min_proxy, "min_overall": min_overall},
    }
