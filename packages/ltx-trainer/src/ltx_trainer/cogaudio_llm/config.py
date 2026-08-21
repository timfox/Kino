"""CogAudio-LLM — cognitive affective reasoning for ALMs (arXiv:2606.06940)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CogAudioLlmConfig:
    paper_arxiv: str = "arXiv:2606.06940"
    title: str = (
        "Beyond Semantic Dominance: Cognitive Affective Reasoning and "
        "Empathetic Response Alignment in Audio Language Models"
    )
    framework: str = "CogAudio-LLM"
    backbone: str = "Qwen2.5-Omni-7B"
    dataset: str = "LIME-440K"
    repo_url: str = "https://github.com/zxzhao0/CogAudio-LLM"

    # LIME-440K (Table 1)
    lime_utterances: int = 438_884
    lime_hours: float = 497.1
    lime_emotions: int = 7
    lime_intensity_levels: int = 3

    # EIPS steps
    eips_steps: tuple[str, ...] = (
        "emotion_perception",
        "intent_extraction",
        "psychological_modeling",
        "strategy_formulation",
    )

    # Training (§4.1)
    lora_rank: int = 8
    lora_alpha: int = 32
    sft_lr: float = 1e-5
    sft_batch: int = 512
    sft_epochs: int = 3
    sapo_steps: int = 1500
    sapo_lr: float = 1e-6
    mixed_prompt_ratio: float = 0.5

    # DR-SAPO Route 1 weights (Eq. 4)
    lambda_fmt: float = 0.1
    lambda_res: float = 0.3
    lambda_emo: float = 0.3
    lambda_intent: float = 0.1
    lambda_psych: float = 0.1
    lambda_strategy: float = 0.1

    # Table 2 — empathy implicit (CogAudio-LLM)
    empathy_esd_conflict: float = 2.90
    empathy_esd_nonconf: float = 2.91
    empathy_humdial_conflict: float = 3.24
    empathy_humdial_nonconf: float = 3.16
    empathy_human_humdial: float = 3.17

    # Table 3 — emotion accuracy full model
    emo_acc_overall: float = 49.5
    emo_acc_conflict: float = 46.0
    emo_acc_nonconf: float = 71.0
