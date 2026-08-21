"""Smoke evaluation (auto-generated; extend with domain-specific toy calls)."""

from __future__ import annotations

from typing import Any


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.holofair.config import HoloFairConfig
    from ltx_trainer.holofair.metrics import mgbi_score
    from ltx_trainer.holofair.pipeline import demo_balanced_counts

    cfg = HoloFairConfig()
    neutral = demo_balanced_counts(cfg=cfg)
    semantic = {t: demo_balanced_counts(cfg=cfg) for t in cfg.semantic_triggers[:3]}
    scores = mgbi_score(neutral, semantic, cfg=cfg)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()}
