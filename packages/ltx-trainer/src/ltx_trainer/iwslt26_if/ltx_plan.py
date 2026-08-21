"""LTX / AV integration for long-form speech IF."""

from __future__ import annotations

from typing import Any

from ltx_trainer.iwslt26_if.config import IWSLT26IFConfig


def ltx_integration_plan(cfg: IWSLT26IFConfig | None = None) -> dict[str, Any]:
    cfg = cfg or IWSLT26IFConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "Long-form clip captions / chapter sidecars for teleplay QA",
            "Multilingual prompt templates aligned with IWSLT IF track",
            "Rerank pool metadata (greedy + SHAS + samples) on exported WAV",
        ],
        "note": "Speech IF on Qwen2.5-Omni; not wired to LTX denoiser — planning stub only",
    }
