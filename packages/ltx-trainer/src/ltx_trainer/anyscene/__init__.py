"""AnyScene controllable driving scene generation (Zhang et al., arXiv:2605.26113)."""

from ltx_trainer.anyscene.bev import BEVLayoutEncoder
from ltx_trainer.anyscene.config import AnySceneConfig
from ltx_trainer.anyscene.ggve import (
    GGVEControlHints,
    SURROUND_VIEWS,
    plucker_embedding,
    render_coordinate_buffer,
    render_semantic_buffer,
    surround_expansion_schedule,
)
from ltx_trainer.anyscene.metrics import bev_topdown, binary_iou, mean_iou
from ltx_trainer.anyscene.occupancy_vae import OccupancyVAE, vae_loss
from ltx_trainer.anyscene.pipeline import (
    evaluation_demo,
    framework_card,
    nucraftv2_curation_stages,
    table_ablation_ggve_buffers,
    table_ablation_stocc_dit,
    table_occupancy_generation_nucraftv2,
    table_occupancy_vae_reconstruction,
    table_video_generation,
)
from ltx_trainer.anyscene.stocc_dit import (
    STOccDiT,
    STOccDiTBlock,
    causal_temporal_mask,
    concat_bev_occ_tokens,
    flow_matching_loss,
    flow_matching_noisy_latent,
)

__all__ = [
    "AnySceneConfig",
    "BEVLayoutEncoder",
    "GGVEControlHints",
    "OccupancyVAE",
    "STOccDiT",
    "STOccDiTBlock",
    "SURROUND_VIEWS",
    "bev_topdown",
    "binary_iou",
    "causal_temporal_mask",
    "concat_bev_occ_tokens",
    "evaluation_demo",
    "flow_matching_loss",
    "flow_matching_noisy_latent",
    "framework_card",
    "mean_iou",
    "nucraftv2_curation_stages",
    "plucker_embedding",
    "render_coordinate_buffer",
    "render_semantic_buffer",
    "surround_expansion_schedule",
    "table_ablation_ggve_buffers",
    "table_ablation_stocc_dit",
    "table_occupancy_generation_nucraftv2",
    "table_occupancy_vae_reconstruction",
    "table_video_generation",
    "vae_loss",
]
