"""Dasheng AudioGen: unified mixed-audio scene generation (arXiv:2605.27838)."""

from ltx_trainer.dasheng_audiogen.captions import (
    CAPTION_VIEWS,
    REQUIRED_VIEW,
    StructuredCaption,
    caption_smoke,
    parse_structured_caption,
)
from ltx_trainer.dasheng_audiogen.config import DashengAudioGenConfig
from ltx_trainer.dasheng_audiogen.eval import eval_smoke, expert_pipeline_beats_unified_on_sma, run_generation_eval
from ltx_trainer.dasheng_audiogen.flow import flow_matching_loss, flow_smoke, linear_interpolate
from ltx_trainer.dasheng_audiogen.metrics_compute import fad_proxy, metrics_compute_smoke, metrics_from_arrays, word_error_rate
from ltx_trainer.dasheng_audiogen.model import FlowMatchingDiT, model_smoke, predict_velocity
from ltx_trainer.dasheng_audiogen.sampler import encode_text_views, sample_flow_matching, sample_to_waveform, sampler_smoke
from ltx_trainer.dasheng_audiogen.tokenizer import decode_latents, encode_waveform, tokenizer_smoke
from ltx_trainer.dasheng_audiogen.layout import LIMITATIONS
from ltx_trainer.dasheng_audiogen.metrics import (
    mecat_category_legend,
    table_1_capability_comparison,
    table_2_standard_benchmarks,
    table_3_mecat_mixed,
    table_4_structured_ablation,
    unified_vs_acoustic_gains,
)
from ltx_trainer.dasheng_audiogen.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
)

__all__ = [
    "CAPTION_VIEWS",
    "DashengAudioGenConfig",
    "LIMITATIONS",
    "REQUIRED_VIEW",
    "StructuredCaption",
    "benchmarks_bundle",
    "caption_smoke",
    "decode_latents",
    "encode_text_views",
    "encode_waveform",
    "eval_smoke",
    "evaluation_demo",
    "expert_pipeline_beats_unified_on_sma",
    "fad_proxy",
    "flow_matching_loss",
    "flow_smoke",
    "FlowMatchingDiT",
    "framework_card",
    "headline_results",
    "linear_interpolate",
    "metrics_compute_smoke",
    "metrics_from_arrays",
    "model_smoke",
    "mecat_category_legend",
    "parse_structured_caption",
    "pipeline_demo",
    "predict_velocity",
    "run_generation_eval",
    "sample_flow_matching",
    "sample_to_waveform",
    "sampler_smoke",
    "table_1_capability_comparison",
    "table_2_standard_benchmarks",
    "table_3_mecat_mixed",
    "table_4_structured_ablation",
    "tokenizer_smoke",
    "unified_vs_acoustic_gains",
    "word_error_rate",
]
