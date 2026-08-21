"""FiLM speaker-conditioned pathological ASR (arXiv:2606.06211)."""

from ltx_trainer.film_spk_asr.config import FilmSpkAsrConfig
from ltx_trainer.film_spk_asr.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.film_spk_asr.film import (
    film_generator,
    gate_alpha,
    gated_film_modulate,
    identity_init_gamma_beta,
    mask_speaker_embedding,
)
from ltx_trainer.film_spk_asr.fold import annotate_audio_save_data
from ltx_trainer.film_spk_asr.mock import evaluation_smoke
from ltx_trainer.film_spk_asr.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_asr_wer,
    table2_mcqa,
)

__all__ = [
    "FilmSpkAsrConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "film_generator",
    "framework_card",
    "gate_alpha",
    "gated_film_modulate",
    "headline_results",
    "identity_init_gamma_beta",
    "mask_speaker_embedding",
    "pipeline_demo",
    "pipeline_demo_export",
    "table1_asr_wer",
    "table2_mcqa",
]
