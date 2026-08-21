"""UniSRM reward smoke (arXiv:2605.23261)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.unisrm.config import UnisrmConfig
from ltx_trainer.unisrm.rewards import accuracy_reward_mos, accuracy_reward_pairwise, format_reward


def evaluation_smoke(cfg: UnisrmConfig | None = None) -> dict[str, Any]:
    c = cfg or UnisrmConfig()
    fmt = format_reward("<think>ok</think><answer>A</answer>")
    mos = accuracy_reward_mos(4.2, 4.0)
    return {
        "paper": c.paper_arxiv,
        "format_reward": fmt,
        "mos_reward": round(mos, 4),
        "r_acc": accuracy_reward_pairwise("A", "A"),
    }
