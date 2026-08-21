"""Augment Engineering pipeline and limitations."""

from __future__ import annotations

PIPELINE_STAGES: tuple[str, ...] = (
    "three_discipline_progression",
    "six_phase_orchestration",
    "portability_metrics",
    "orchestration_patterns",
    "governance_checkpoints",
)

LIMITATIONS: tuple[str, ...] = (
    "Single-practitioner formative case study — not multi-site confirmatory evidence.",
    "No reproduction of 200-interaction corpus or 82-artifact Wright's Law fit on real data.",
    "HeyGen/Gamma excluded from numeric aggregates per paper instrumentation limits.",
    "Cochran-Armitage and Wright's Law smokes use paper-reported contingency tables.",
)
