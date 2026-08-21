"""Rubric refinement actions from dual-loop triggers (GrowLoop §3.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.growloop.config import GrowLoopConfig
from ltx_trainer.growloop.dual_loop import TriggerCell, anchor_drift_ok

_DEFAULT_DIMS = (
    "coherence",
    "helpfulness",
    "correctness",
    "clarity",
    "depth",
    "creativity",
    "instruction_following",
    "harmlessness",
    "empathy",
    "conciseness",
    "factuality",
    "reasoning",
    "formatting",
    "tone",
    "completeness",
    "engagement",
    "tool_use",
    "multilingual",
)


def quality_dimension_names(cfg: GrowLoopConfig | None = None) -> list[str]:
    cfg = cfg or GrowLoopConfig()
    n = cfg.quality_dimensions
    if n <= len(_DEFAULT_DIMS):
        return list(_DEFAULT_DIMS[:n])
    return list(_DEFAULT_DIMS) + [f"dimension_{i}" for i in range(len(_DEFAULT_DIMS), n)]


@dataclass
class RubricState:
    version: int
    dimensions: list[str]
    anchors: dict[str, float] = field(default_factory=dict)


def default_rubric(cfg: GrowLoopConfig | None = None) -> RubricState:
    cfg = cfg or GrowLoopConfig()
    return RubricState(
        version=1,
        dimensions=quality_dimension_names(cfg),
        anchors={d: 3.5 for d in quality_dimension_names(cfg)[:3]},
    )


def apply_trigger(rubric: RubricState, trigger: TriggerCell) -> RubricState:
    dims = list(rubric.dimensions)
    anchors = dict(rubric.anchors)
    if trigger == TriggerCell.RUBRIC_QUALITY:
        for k in list(anchors):
            anchors[k] = min(5.0, anchors[k] + 0.05)
    elif trigger == TriggerCell.RUBRIC_COVERAGE:
        dims.append("novel_failure_pattern")
        anchors["novel_failure_pattern"] = 2.5
    elif trigger == TriggerCell.CASE_DISCRIMINABILITY:
        for k in list(anchors):
            anchors[k] = max(1.0, anchors[k] - 0.02)
    elif trigger == TriggerCell.CASE_SATURATION:
        dims.append("difficulty_ceiling")
        anchors["difficulty_ceiling"] = 4.0
    return RubricState(version=rubric.version + 1, dimensions=dims, anchors=anchors)


def refine_rubric(
    rubric: RubricState,
    trigger: TriggerCell,
    *,
    baseline_scores: list[float],
    cfg: GrowLoopConfig | None = None,
) -> RubricState | None:
    """Reject update if frozen anchors drift too far."""
    cfg = cfg or GrowLoopConfig()
    candidate = apply_trigger(rubric, trigger)
    new_scores = [candidate.anchors.get(d, 3.0) for d in rubric.dimensions[: len(baseline_scores)]]
    if not anchor_drift_ok(baseline_scores, new_scores, epsilon=cfg.anchor_drift_epsilon if hasattr(cfg, "anchor_drift_epsilon") else 2.0):
        return None
    return candidate


def rubric_editor_smoke() -> dict[str, Any]:
    rub = default_rubric()
    updated = apply_trigger(rub, TriggerCell.RUBRIC_COVERAGE)
    ok = refine_rubric(rub, TriggerCell.RUBRIC_QUALITY, baseline_scores=[3.5, 3.5, 3.5])
    return {
        "initial_dims": len(rub.dimensions),
        "after_coverage_dims": len(updated.dimensions),
        "refine_accepted": ok is not None,
        "version_bump": updated.version > rub.version,
    }
