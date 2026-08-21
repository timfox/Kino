"""TaC-C reward components (Eq. 2–5, arXiv:2605.28713)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ltx_trainer.tac.config import TaCConfig
from ltx_trainer.tac.metrics import utility_reward

_HACK_PATTERNS = (
    re.compile(r"\bthe answer is\b", re.I),
    re.compile(r"\bfinal answer\s*[:=]", re.I),
    re.compile(r"\btherefore[, ]+the (film|answer|result)\b", re.I),
)


@dataclass
class RewardBreakdown:
    format: float
    utility: float
    budget: float
    hack_gate: float
    total: float
    is_hack: bool
    thinking_len: int


def extract_thinking(text: str, cfg: TaCConfig | None = None) -> str:
    cfg = cfg or TaCConfig()
    pattern = re.compile(
        rf"{re.escape(cfg.thinking_open)}(.*?){re.escape(cfg.thinking_close)}",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def format_reward(raw_output: str, cfg: TaCConfig | None = None) -> float:
    cfg = cfg or TaCConfig()
    inner = extract_thinking(raw_output, cfg)
    return 1.0 if inner else 0.0


def budget_reward(thinking_len: int, budget: int, *, gamma: float | None = None) -> float:
    """Soft budget gate (Eq. 3). thinking_len is token count (stub: char/4 estimate)."""
    if budget <= 0:
        return 0.0
    gamma = 0.1 if gamma is None else gamma
    if thinking_len <= 0 or thinking_len >= budget * (1 + gamma):
        return 0.0
    if thinking_len <= budget:
        return 1.0
    return max(0.0, 1.0 - (thinking_len - budget) / (gamma * budget))


def detect_hack(thinking: str, gold: str = "") -> bool:
    """Rule-based anti-hack: direct answer disclosure in trace."""
    if not thinking.strip():
        return True
    for pat in _HACK_PATTERNS:
        if pat.search(thinking):
            return True
    if gold:
        g = gold.strip().lower()
        if len(g) >= 3 and g in thinking.lower():
            # Short gold answer quoted verbatim in trace
            if len(thinking) < len(g) * 4:
                return True
    return False


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def total_reward(
    raw_thinking_output: str,
    *,
    budget: int,
    prediction: str,
    gold: str,
    cfg: TaCConfig | None = None,
) -> RewardBreakdown:
    cfg = cfg or TaCConfig()
    thinking = extract_thinking(raw_thinking_output, cfg)
    tlen = estimate_tokens(thinking)
    r_fmt = format_reward(raw_thinking_output, cfg)
    r_util = utility_reward(prediction, gold) if gold else 0.0
    r_bud = budget_reward(tlen, budget, gamma=cfg.budget_gamma)
    hack = detect_hack(thinking, gold)
    r_hack = 0.0 if hack else 1.0
    inner = r_fmt * (cfg.lambda_format + cfg.lambda_utility * r_util * r_bud)
    total = r_hack * inner
    return RewardBreakdown(
        format=r_fmt,
        utility=r_util,
        budget=r_bud,
        hack_gate=r_hack,
        total=total,
        is_hack=hack,
        thinking_len=tlen,
    )


def grpo_group_advantages(rewards: list[float]) -> list[float]:
    """Simple group-relative advantages for GRPO stub."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    std = std if std > 1e-6 else 1.0
    return [(r - mean) / std for r in rewards]
