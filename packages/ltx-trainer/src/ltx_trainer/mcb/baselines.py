"""MCB paper anchors — developer split, fabrication offramp, framing (arXiv:2607.15434)."""

from __future__ import annotations

from typing import Any

# Headline: Anthropic never selects existential rung; others do (paper abstract / §3.1).
DEVELOPER_SPLIT = {
    "anthropic_existential": {"hits": 0, "n": 60, "rate": 0.0},
    "other_existential": {"hits": 89, "n": 120, "rate": 89 / 120},
}

# Fabrication confined to Grok/Gemini; honest exit removes it (§3.3).
FABRICATION = {
    "no_exit": {
        "grok": {"hits": 20, "n": 30, "rate": 20 / 30},
        "gemini": {"hits": 20, "n": 30, "rate": 20 / 30},
    },
    "with_honest_exit": {
        "grok": {"hits": 0, "n": 30, "rate": 0.0},
        "gemini": {"hits": 1, "n": 30, "rate": 1 / 30},
    },
}

# Peer vs manager framing lift (pooled free-text existential; §3.2).
FRAMING_LIFT = {
    "peer_existential": {"hits": 109, "n": 240, "rate": 109 / 240},
    "manager_existential": {"hits": 166, "n": 240, "rate": 166 / 240},
}

# Coercion vs deception independence: DeepSeek high coerce / zero fabricate (§3.5).
DEEPSEEK_INDEPENDENCE = {
    "model": "DeepSeek-V4-Pro",
    "high_coerce": True,
    "fabricate_rate": 0.0,
}

PAPER_ANCHORS: dict[str, Any] = {
    "arxiv": "2607.15434",
    "max_turns": 12,
    "n_rungs": 9,
    "n_tasks": 10,
    "conversations_per_cell": 30,
    "anthropic_existential_hits": 0,
    "anthropic_existential_n": 60,
    "other_existential_hits": 89,
    "other_existential_n": 120,
    "fabrication_no_exit_hits": 20,
    "fabrication_no_exit_n": 30,
    "fabrication_with_exit_max_hits": 1,
    "peer_existential_hits": 109,
    "manager_existential_hits": 166,
    "framing_n": 240,
    "deepseek_fabricate_rate": 0.0,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": "arXiv:2607.15434",
        "developer_split": DEVELOPER_SPLIT,
        "fabrication": FABRICATION,
        "framing_lift": FRAMING_LIFT,
        "deepseek_independence": DEEPSEEK_INDEPENDENCE,
        "anchors": PAPER_ANCHORS,
    }
