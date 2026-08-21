"""SSL soft discrete-token inference (Onda et al., arXiv:2606.06806)."""

from ltx_trainer.ssl_soft_infer.config import AssignmentMode, SslModel, SslSoftInferConfig
from ltx_trainer.ssl_soft_infer.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.ssl_soft_infer.fold import annotate_audio_save_data
from ltx_trainer.ssl_soft_infer.mock import evaluation_smoke
from ltx_trainer.ssl_soft_infer.phoneme import phoneme_separability_demo
from ltx_trainer.ssl_soft_infer.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    experimental_protocol,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_asr_wer,
    table2_synthesis,
    table3_phoneme_separability,
    table4_multilayer_asr,
)
from ltx_trainer.ssl_soft_infer.posterior import (
    hard_token,
    posterior_demo,
    soft_posterior,
    squared_distances,
    tau_for_dataset,
    token_embedding,
)

__all__ = [
    "AssignmentMode",
    "SslModel",
    "SslSoftInferConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "experimental_protocol",
    "framework_card",
    "hard_token",
    "headline_results",
    "phoneme_separability_demo",
    "pipeline_demo",
    "pipeline_demo_export",
    "posterior_demo",
    "soft_posterior",
    "squared_distances",
    "table1_asr_wer",
    "table2_synthesis",
    "table3_phoneme_separability",
    "table4_multilayer_asr",
    "tau_for_dataset",
    "token_embedding",
]
