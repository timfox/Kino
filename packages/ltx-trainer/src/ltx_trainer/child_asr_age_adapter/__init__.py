"""Age-aware adapter tuning for child ASR (arXiv:2606.05440)."""

from ltx_trainer.child_asr_age_adapter.adapters import (
    age_router_logits,
    bottleneck_adapter,
    combine_encoder_representations,
    film_modulate,
    route_top_k,
    transducer_nll,
)
from ltx_trainer.child_asr_age_adapter.config import ChildAsrAgeAdapterConfig
from ltx_trainer.child_asr_age_adapter.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.child_asr_age_adapter.fold import annotate_audio_save_data
from ltx_trainer.child_asr_age_adapter.mock import evaluation_smoke
from ltx_trainer.child_asr_age_adapter.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_dataset_stats,
    table2_group_wer,
    table2_wer_results,
)

__all__ = [
    "ChildAsrAgeAdapterConfig",
    "age_router_logits",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "bottleneck_adapter",
    "combine_encoder_representations",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "film_modulate",
    "framework_card",
    "headline_results",
    "pipeline_demo",
    "pipeline_demo_export",
    "route_top_k",
    "table1_dataset_stats",
    "table2_group_wer",
    "table2_wer_results",
    "transducer_nll",
]
