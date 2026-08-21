"""SwanSphere — streaming synchronized spatial audio (arXiv:2605.30940)."""

from ltx_trainer.swansphere.config import SwanSphereConfig
from ltx_trainer.swansphere.foa import FOA_CHANNEL_NAMES, intensity_vector_azimuth, pseudo_foa_from_stereo
from ltx_trainer.swansphere.layout import LIMITATIONS
from ltx_trainer.swansphere.odpo import odpo_reward, rank_candidates
from ltx_trainer.swansphere.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_summary,
    pipeline_demo,
    table_model_ablation,
    table_svac_ablation,
    table_text_to_foa,
    table_video_to_foa,
    table_wcs,
)
from ltx_trainer.swansphere.streaming import latency_breakdown, plan_patches, stream_patch_latents
from ltx_trainer.swansphere.svac import svac_batch_loss

__all__ = [
    "FOA_CHANNEL_NAMES",
    "LIMITATIONS",
    "SwanSphereConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "framework_card",
    "intensity_vector_azimuth",
    "knowledge_summary",
    "latency_breakdown",
    "odpo_reward",
    "pipeline_demo",
    "plan_patches",
    "pseudo_foa_from_stereo",
    "rank_candidates",
    "stream_patch_latents",
    "svac_batch_loss",
    "table_model_ablation",
    "table_svac_ablation",
    "table_text_to_foa",
    "table_video_to_foa",
    "table_wcs",
]
