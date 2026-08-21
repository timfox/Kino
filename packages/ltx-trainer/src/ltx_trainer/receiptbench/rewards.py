"""Metric-aware GRPO reward shaping (Sec. 4.2, Eq. 2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.receiptbench.config import ReceiptBenchConfig
from ltx_trainer.receiptbench.evaluation import field_similarity
from ltx_trainer.receiptbench.schema import ALL_FIELDS


def field_reward(
    pred: Any,
    gold: Any,
    field: str,
    *,
    cfg: ReceiptBenchConfig | None = None,
) -> float:
    """R_f from confusion matrix states (Eq. 2)."""
    cfg = cfg or ReceiptBenchConfig()

    def _empty(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            return v.strip() == ""
        if isinstance(v, list):
            return len(v) == 0
        return False

    pe, ge = _empty(pred), _empty(gold)
    s = field_similarity(pred, gold, field, cfg=cfg)

    if not ge and not pe:
        return s
    if ge and pe:
        return cfg.reward_tn
    if ge and not pe:
        return cfg.reward_fp
    return cfg.reward_fn


def invoice_reward(
    pred: dict[str, Any],
    gold: dict[str, Any],
    *,
    cfg: ReceiptBenchConfig | None = None,
) -> float:
    """Average reward across 19 fields."""
    cfg = cfg or ReceiptBenchConfig()
    rewards = [field_reward(pred.get(f), gold.get(f), f, cfg=cfg) for f in ALL_FIELDS]
    return sum(rewards) / len(rewards) if rewards else 0.0
