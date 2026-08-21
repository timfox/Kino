"""Conversation Situation Package (CSP) — GrowLoop Table 2."""

from __future__ import annotations

from typing import Any

from ltx_trainer.growloop.config import GrowLoopConfig

CSP_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "group": "Context",
        "fields": ("scene", "topic", "relationship", "social_expectation"),
    },
    {
        "group": "User",
        "fields": ("persona", "intent", "emotional_state"),
    },
    {
        "group": "Conversation",
        "fields": ("ambiguity_pattern", "turn_structure"),
    },
    {
        "group": "Test plan",
        "fields": ("difficulty", "failure_trigger", "target_rubric_dims"),
    },
    {
        "group": "Cognition (optional)",
        "fields": ("challenge_type", "variant", "failure_mode"),
        "optional": True,
    },
)


def csp_field_registry(cfg: GrowLoopConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or GrowLoopConfig()
    rows: list[dict[str, Any]] = []
    for g in CSP_GROUPS:
        for f in g["fields"]:
            rows.append(
                {
                    "group": g["group"],
                    "field": f,
                    "optional_group": g.get("optional", False),
                }
            )
    assert len(rows) == cfg.csp_fields
    return rows


def csp_pool_sources(cfg: GrowLoopConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    return {
        "real_conversations": cfg.real_conversations,
        "real_user_messages": cfg.real_user_messages,
        "seed_conversations": cfg.seed_cases,
        "trap_taxonomy_categories": cfg.trap_categories,
        "rubric_dimensions": cfg.quality_dimensions,
    }
