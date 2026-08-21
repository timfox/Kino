"""AAC with auxiliary AudioSet semantics (arXiv:2606.05717)."""

from ltx_trainer.aac_audioset.config import AacAudiosetConfig
from ltx_trainer.aac_audioset.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.aac_audioset.fold import annotate_audio_save_data
from ltx_trainer.aac_audioset.fusion import (
    caption_ce_loss,
    decode_step_argmax,
    fuse_acoustic_semantic,
    top_k_audioset_keywords,
)
from ltx_trainer.aac_audioset.mock import evaluation_smoke
from ltx_trainer.aac_audioset.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_clotho,
    table2_audiocaps,
    table3_keyword_ablation,
    table4_k_sensitivity,
)

__all__ = [
    "AacAudiosetConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "caption_ce_loss",
    "decode_step_argmax",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "fuse_acoustic_semantic",
    "headline_results",
    "pipeline_demo",
    "pipeline_demo_export",
    "table1_clotho",
    "table2_audiocaps",
    "table3_keyword_ablation",
    "table4_k_sensitivity",
    "top_k_audioset_keywords",
]
