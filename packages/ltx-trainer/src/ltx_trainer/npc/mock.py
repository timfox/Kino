"""NPC finite-blocklength rate bounds smoke (arXiv:2605.25699)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.npc.bounds import log_m_star_achievability_lower, log_m_star_converse_upper
from ltx_trainer.npc.config import NPCConfig


def evaluation_smoke(cfg: NPCConfig | None = None) -> dict[str, Any]:
    c = cfg or NPCConfig()
    upper = log_m_star_converse_upper(n=100, d=4, lambda_star=0.5)
    lower = log_m_star_achievability_lower(n=100, d=4, c=1.0, lambda_star=0.5)
    return {
        "paper": "arXiv:2605.25699",
        "log_m_upper": round(upper, 3),
        "log_m_lower": round(lower, 3),
    }
