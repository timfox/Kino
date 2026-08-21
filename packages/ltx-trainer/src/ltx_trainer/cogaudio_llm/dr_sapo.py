"""DR-SAPO dual-route reward routing (§3.3)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cogaudio_llm.config import CogAudioLlmConfig


def route_reward(
    *,
    prompt_a: bool,
    fmt_score: float,
    emo_score: float,
    intent_score: float,
    psych_score: float,
    strategy_score: float,
    empathy_score: float,
    cfg: CogAudioLlmConfig | None = None,
) -> float:
    """R(o) = I{PA}·RF1 + I{PB}·RF2 (Eq. 3–4)."""
    cfg = cfg or CogAudioLlmConfig()
    if prompt_a:
        rf1 = (
            cfg.lambda_fmt * fmt_score
            + cfg.lambda_emo * emo_score
            + cfg.lambda_intent * intent_score
            + cfg.lambda_psych * psych_score
            + cfg.lambda_strategy * strategy_score
            + cfg.lambda_res * empathy_score
        )
        return rf1
    return cfg.lambda_res * empathy_score


def dr_sapo_demo(*, cfg: CogAudioLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CogAudioLlmConfig()
    rf1 = route_reward(
        prompt_a=True,
        fmt_score=1.0,
        emo_score=0.9,
        intent_score=0.85,
        psych_score=0.8,
        strategy_score=0.85,
        empathy_score=0.92,
        cfg=cfg,
    )
    rf2 = route_reward(
        prompt_a=False,
        fmt_score=0.0,
        emo_score=0.0,
        intent_score=0.0,
        psych_score=0.0,
        strategy_score=0.0,
        empathy_score=0.92,
        cfg=cfg,
    )
    return {
        "route1_explicit_rf1": round(rf1, 4),
        "route2_implicit_rf2": round(rf2, 4),
        "lambda_emo": cfg.lambda_emo,
        "lambda_res": cfg.lambda_res,
        "sapo_steps": cfg.sapo_steps,
    }
