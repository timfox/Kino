"""CogAudio-LLM — cognitive affective reasoning for ALMs (arXiv:2606.06940)."""

from ltx_trainer.cogaudio_llm.config import CogAudioLlmConfig
from ltx_trainer.cogaudio_llm.dr_sapo import dr_sapo_demo, route_reward
from ltx_trainer.cogaudio_llm.eips import eips_chain, eips_demo, format_cot_output
from ltx_trainer.cogaudio_llm.eval import eval_smoke, pipeline_demo
from ltx_trainer.cogaudio_llm.fold import annotate_audio_save_data
from ltx_trainer.cogaudio_llm.lime import decouple_text_emotions, lime_demo, lime_statistics
from ltx_trainer.cogaudio_llm.mock import evaluation_smoke
from ltx_trainer.cogaudio_llm.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table2_empathy_quality,
    table3_emotion_accuracy,
    table4_empathy_ablation,
)

__all__ = [
    "CogAudioLlmConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "decouple_text_emotions",
    "dr_sapo_demo",
    "eips_chain",
    "eips_demo",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "format_cot_output",
    "framework_card",
    "headline_results",
    "lime_demo",
    "lime_statistics",
    "pipeline_demo",
    "route_reward",
    "table2_empathy_quality",
    "table3_emotion_accuracy",
    "table4_empathy_ablation",
]
