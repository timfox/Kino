"""AudioIM — timbre/tempo-controllable V2A with reference audio (arXiv:2606.07182)."""

from ltx_trainer.audioim.config import AudioImConfig
from ltx_trainer.audioim.eval import eval_smoke, pipeline_demo
from ltx_trainer.audioim.fold import annotate_audio_save_data
from ltx_trainer.audioim.masking import flow_matching_loss, masking_demo, split_latent_mask
from ltx_trainer.audioim.mock import evaluation_smoke
from ltx_trainer.audioim.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_v2a_performance,
    table2_style_similarity,
)
from ltx_trainer.audioim.style import encode_tempo, encode_timbre, fuse_style, style_conditioning_demo

__all__ = [
    "AudioImConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "encode_tempo",
    "encode_timbre",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "flow_matching_loss",
    "framework_card",
    "fuse_style",
    "headline_results",
    "masking_demo",
    "pipeline_demo",
    "split_latent_mask",
    "style_conditioning_demo",
    "table1_v2a_performance",
    "table2_style_similarity",
]
