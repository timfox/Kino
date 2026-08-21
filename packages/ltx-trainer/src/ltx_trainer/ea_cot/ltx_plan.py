"""LTX / agent hooks for speech reasoning diagnostics."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ea_cot.config import EACoTConfig


def ltx_integration_plan(cfg: EACoTConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EACoTConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "Pre-flight SLLM prompt: EA-CoT on entity-heavy teleplay / dialogue shots",
            "Compare cascaded ASR+Gemma vs end-to-end Omni on logic-heavy agent tools",
            "VoiceBench-style regression before shipping long-form speech IF LoRAs",
        ],
        "note": "Inference-time prompting only; no fine-tune in Gopex stub",
    }
