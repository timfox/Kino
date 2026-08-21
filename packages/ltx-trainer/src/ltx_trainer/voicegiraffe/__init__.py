"""VOICEGIRAFFE: extreme long-context audio-language benchmark (arXiv:2605.27976)."""

from ltx_trainer.voicegiraffe.config import (
    Domain,
    InferenceMode,
    MultiHopTask,
    SingleHopTask,
    TaskTier,
    VoiceGiraffeConfig,
)
from ltx_trainer.voicegiraffe.cascade import aggregate_captions, cascade_evaluate, cascade_predict, sliding_window_segments
from ltx_trainer.voicegiraffe.dataset import QAItem, items_to_json, load_qa_items
from ltx_trainer.voicegiraffe.eval import (
    duration_decay_curve,
    eval_smoke,
    extract_mc_answer,
    inference_mode_scores,
    mc_accuracy,
    run_eval,
)
from ltx_trainer.voicegiraffe.recording_registry import (
    build_recording_catalog,
    build_synthetic_qa_pool,
    catalog_summary,
    load_full_qa_pool,
    recording_registry_smoke,
)
from ltx_trainer.voicegiraffe.hub_audio import (
    build_audio_manifest,
    export_audio_manifest,
    hub_audio_smoke,
    load_audio_manifest,
    load_recording_waves,
    synthesize_recording_wave,
)
from ltx_trainer.voicegiraffe.hub_manifest import (
    build_hub_manifest,
    export_hub_bundle,
    hub_dataset_card,
    hub_manifest_smoke,
)
from ltx_trainer.voicegiraffe.hub_loader import (
    export_qa_jsonl,
    export_recording_manifest,
    hub_loader_smoke,
    load_qa_jsonl,
    load_recording_manifest,
    recording_catalog,
)
from ltx_trainer.voicegiraffe.pool_eval import pool_eval_smoke, run_sampled_pool_eval, stratified_sample
from ltx_trainer.voicegiraffe.lrm import lrm_evaluate, lrm_predict
from ltx_trainer.voicegiraffe.layout import LIMITATIONS
from ltx_trainer.voicegiraffe.metrics import (
    memory_asymmetry,
    table_1_benchmark_comparison,
    table_2_leaderboard,
    table_3_lrm_ablation,
    table_4_language_bias,
)
from ltx_trainer.voicegiraffe.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
)
from ltx_trainer.voicegiraffe.taxonomy import benchmark_stats, domain_coverage, task_taxonomy

__all__ = [
    "Domain",
    "InferenceMode",
    "LIMITATIONS",
    "MultiHopTask",
    "SingleHopTask",
    "TaskTier",
    "VoiceGiraffeConfig",
    "QAItem",
    "aggregate_captions",
    "benchmark_stats",
    "benchmarks_bundle",
    "build_audio_manifest",
    "build_recording_catalog",
    "build_synthetic_qa_pool",
    "cascade_evaluate",
    "cascade_predict",
    "domain_coverage",
    "duration_decay_curve",
    "eval_smoke",
    "evaluation_demo",
    "extract_mc_answer",
    "build_hub_manifest",
    "export_hub_bundle",
    "hub_dataset_card",
    "hub_manifest_smoke",
    "export_qa_jsonl",
    "export_audio_manifest",
    "export_recording_manifest",
    "framework_card",
    "headline_results",
    "hub_audio_smoke",
    "hub_loader_smoke",
    "inference_mode_scores",
    "items_to_json",
    "load_audio_manifest",
    "load_full_qa_pool",
    "load_qa_items",
    "load_qa_jsonl",
    "load_recording_manifest",
    "load_recording_waves",
    "lrm_evaluate",
    "lrm_predict",
    "mc_accuracy",
    "memory_asymmetry",
    "pipeline_demo",
    "pool_eval_smoke",
    "recording_catalog",
    "recording_registry_smoke",
    "run_eval",
    "run_sampled_pool_eval",
    "sliding_window_segments",
    "stratified_sample",
    "synthesize_recording_wave",
    "table_1_benchmark_comparison",
    "table_2_leaderboard",
    "table_3_lrm_ablation",
    "table_4_language_bias",
    "task_taxonomy",
    "catalog_summary",
]
