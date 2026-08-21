"""F5-TTS-DPS: EMA + dual-scoring prompt selection for WildSpoof TTS."""

from ltx_trainer.f5_tts_dps.config import F5TtsDpsConfig
from ltx_trainer.f5_tts_dps.dps import (
    AUDIO_SCORING_PROMPT_HEAD,
    TEXT_SCORING_PROMPT_HEAD,
    audio_expressiveness_score_stub,
    dual_score_filter,
    text_alignment_pick_stub,
)
from ltx_trainer.f5_tts_dps.ema import ema_distance, ema_update
from ltx_trainer.f5_tts_dps.layout import LIMITATIONS
from ltx_trainer.f5_tts_dps.mock import evaluation_smoke
from ltx_trainer.f5_tts_dps.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table_i_dev_ablation,
    table_ii_test_seen_speakers,
)

__all__ = [
    "AUDIO_SCORING_PROMPT_HEAD",
    "F5TtsDpsConfig",
    "LIMITATIONS",
    "TEXT_SCORING_PROMPT_HEAD",
    "audio_expressiveness_score_stub",
    "benchmarks_bundle",
    "dual_score_filter",
    "ema_distance",
    "ema_update",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "table_i_dev_ablation",
    "table_ii_test_seen_speakers",
    "text_alignment_pick_stub",
]
