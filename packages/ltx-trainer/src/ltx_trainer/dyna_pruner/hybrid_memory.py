"""Hybrid-memory turn importance for vLLM chat context compression."""

from __future__ import annotations

import re
from typing import Any


def turn_importance_score(message: dict[str, Any]) -> float:
    """
    Cheap proxy for informational density of a chat turn (higher → keep in CSA).

    Used when ``GOPEX_DYNA_PRUNER=1`` to compress low-value turns into HCA first.
    """
    role = str(message.get("role") or "")
    content = message.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        text = "\n".join(parts)
    text = text.strip()
    if not text:
        return 0.0
    tokens = re.findall(r"\w+", text.lower())
    uniq = len(set(tokens))
    length = len(tokens)
    question_boost = 1.5 if "?" in text else 1.0
    code_boost = 1.3 if "```" in text or "def " in text else 1.0
    role_boost = 1.2 if role == "user" else 1.0
    return float(length + 0.5 * uniq) * question_boost * code_boost * role_boost


def rank_turn_pairs_by_importance(pairs: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """Sort turn pairs ascending by importance (low first → compress to HCA)."""
    return sorted(pairs, key=lambda pair: sum(turn_importance_score(m) for m in pair))
