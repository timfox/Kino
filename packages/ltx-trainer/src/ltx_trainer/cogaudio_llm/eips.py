"""EIPS 4-step Chain-of-Thought reasoning (§2.2, §3.1)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cogaudio_llm.config import CogAudioLlmConfig


def eips_chain(
    *,
    perception: str,
    intent: str,
    psychology: str,
    strategy: str,
    response: str,
) -> dict[str, str]:
    return {
        "step1_perception": perception,
        "step2_intent": intent,
        "step3_psychology": psychology,
        "step4_strategy": strategy,
        "response": response,
    }


def format_cot_output(chain: dict[str, str]) -> str:
    thought = "\n".join(
        [
            f"[Perception] {chain['step1_perception']}",
            f"[Intent] {chain['step2_intent']}",
            f"[Psychology] {chain['step3_psychology']}",
            f"[Strategy] {chain['step4_strategy']}",
        ]
    )
    return f"<thought>{thought}</thought><response>{chain['response']}</response>"


def eips_demo(*, cfg: CogAudioLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CogAudioLlmConfig()
    # Sarcasm / semantic conflict example from Fig. 1
    chain = eips_chain(
        perception="Disgust / sarcasm; flat prosody contradicts positive words",
        intent="Expressing frustration, not celebration",
        psychology="User expects validation of feelings, not cheerleading",
        strategy="Apology and inquiry; avoid enthusiastic echo",
        response="I sense you are frustrated. What went wrong with the outcome?",
    )
    return {
        "steps": list(cfg.eips_steps),
        "num_steps": len(cfg.eips_steps),
        "formatted": format_cot_output(chain),
        "resolves_modality_gap": True,
    }
