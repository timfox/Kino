"""Survey stub: Audio Reasoning in Multimodal Foundation Models (arXiv:2605.21008)."""

from ltx_trainer.audio_reasoning_survey.config import AudioReasoningSurveyConfig
from ltx_trainer.audio_reasoning_survey.formulation import (
    audio_to_text_factorization_steps,
    joint_factorization_exists,
    sequential_audio_to_speech_factorization_steps,
)
from ltx_trainer.audio_reasoning_survey.layout import LIMITATIONS
from ltx_trainer.audio_reasoning_survey.mock import evaluation_smoke
from ltx_trainer.audio_reasoning_survey.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table2_rl_reward_types_excerpt,
    table3_audio_to_text_data_excerpt,
    table5_agent_tradeoffs,
    table6_slm_benchmarks_excerpt,
)
from ltx_trainer.audio_reasoning_survey.taxonomy import agentic_design_patterns, four_paradigms

__all__ = [
    "LIMITATIONS",
    "AudioReasoningSurveyConfig",
    "agentic_design_patterns",
    "audio_to_text_factorization_steps",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "four_paradigms",
    "framework_card",
    "headline_results",
    "joint_factorization_exists",
    "sequential_audio_to_speech_factorization_steps",
    "table2_rl_reward_types_excerpt",
    "table3_audio_to_text_data_excerpt",
    "table5_agent_tradeoffs",
    "table6_slm_benchmarks_excerpt",
]
