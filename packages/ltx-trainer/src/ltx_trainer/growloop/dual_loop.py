"""Rubric–Case dual-loop co-evolution (GrowLoop §3.5, Table 4)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ltx_trainer.growloop.config import GrowLoopConfig


class TriggerCell(str, Enum):
    RUBRIC_QUALITY = "rubric_quality"
    RUBRIC_COVERAGE = "rubric_coverage"
    CASE_DISCRIMINABILITY = "case_discriminability"
    CASE_SATURATION = "case_saturation"


DUAL_LOOP_TRIGGERS: tuple[dict[str, Any], ...] = (
    {
        "loop": "R→I",
        "signal": "dimension consistency ↓",
        "gap": "quality",
        "action": "refine anchors",
        "human_required": False,
    },
    {
        "loop": "R→I",
        "signal": "novel failure pattern",
        "gap": "coverage",
        "action": "add dimension*",
        "human_required": True,
    },
    {
        "loop": "I→R",
        "signal": "case discriminability ↓",
        "gap": "quality",
        "action": "regenerate case",
        "human_required": False,
    },
    {
        "loop": "I→R",
        "signal": "strongest model saturates",
        "gap": "coverage",
        "action": "raise difficulty",
        "human_required": False,
    },
)


def select_trigger(
    *,
    dim_consistency: float,
    novel_failure: bool,
    discriminability: float,
    model_saturated: bool,
    consistency_threshold: float = 0.85,
    delta_threshold: float = 0.32,
) -> TriggerCell | None:
    if novel_failure:
        return TriggerCell.RUBRIC_COVERAGE
    if dim_consistency < consistency_threshold:
        return TriggerCell.RUBRIC_QUALITY
    if model_saturated:
        return TriggerCell.CASE_SATURATION
    if discriminability < delta_threshold:
        return TriggerCell.CASE_DISCRIMINABILITY
    return None


def anchor_drift_ok(
    baseline_scores: list[float],
    new_scores: list[float],
    *,
    epsilon: float = 2.0,
) -> bool:
    """Frozen anchor subset: reject rubric update if scores deviate > ε."""
    if len(baseline_scores) != len(new_scores):
        return False
    return all(abs(a - b) <= epsilon for a, b in zip(baseline_scores, new_scores, strict=True))


def dual_loop_smoke(cfg: GrowLoopConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    t1 = select_trigger(dim_consistency=0.80, novel_failure=False, discriminability=0.35, model_saturated=False)
    t2 = select_trigger(dim_consistency=0.90, novel_failure=True, discriminability=0.40, model_saturated=False)
    return {
        "num_triggers": len(DUAL_LOOP_TRIGGERS),
        "human_required_cells": sum(1 for t in DUAL_LOOP_TRIGGERS if t["human_required"]),
        "low_consistency_trigger": t1 == TriggerCell.RUBRIC_QUALITY,
        "novel_failure_trigger": t2 == TriggerCell.RUBRIC_COVERAGE,
        "anchor_drift_ok": anchor_drift_ok([69.5, 58.0], [70.0, 57.5], epsilon=2.0),
        "case_count_anchor": cfg.case_count,
    }
