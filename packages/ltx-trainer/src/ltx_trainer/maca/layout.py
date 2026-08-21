"""Small textual layout helpers for MACA (arXiv:2605.25746)."""

from __future__ import annotations

from typing import Any


def architecture_layout() -> dict[str, Any]:
    return {
        "GraphSpec": ["Z_prior (agent relevance)", "P_prior (interaction plausibility)", "hard mask H", "π_prior"],
        "Orchestration": ["policy πθ over agents+STOP", "masking (Eq. 11)", "π_mix anchor (Eq. 12)", "GRPO update"],
        "Budget": ["budget_tokens", "token-cost penalty", "temperature β(b)"],
    }


def paper_limitations() -> list[str]:
    return [
        "This repo includes reference GraphSpec + token-aware orchestration demos, not full benchmark training.",
        "We do not ship the paper’s sentence encoder or RL training pipeline; embeddings are deterministic hashes.",
        "Token costs and task success are simulated (for reproducible unit tests).",
    ]

