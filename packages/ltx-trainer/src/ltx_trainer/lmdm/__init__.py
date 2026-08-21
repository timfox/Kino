"""Live Music Diffusion Models (LMDM) — interactive streaming audio diffusion (arXiv:2605.22717)."""

from ltx_trainer.lmdm.arc_forcing import arc_forcing_total, contrastive_loss, relativistic_loss
from ltx_trainer.lmdm.complexity import lmdm_block_causal_ops, lmdm_enc_dec_ops, lmm_decode_ops
from ltx_trainer.lmdm.config import LMDMConfig, LMDMLatencyStats, LMDMVariant
from ltx_trainer.lmdm.layout import LIMITATIONS
from ltx_trainer.lmdm.pipeline import (
    design_space_axes,
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table_sketch_musdb,
    table_text_global,
)
from ltx_trainer.lmdm.routing import routing_mask

__all__ = [
    "LMDMConfig",
    "LMDMLatencyStats",
    "LMDMVariant",
    "LIMITATIONS",
    "arc_forcing_total",
    "contrastive_loss",
    "design_space_axes",
    "evaluation_demo",
    "framework_card",
    "lmdm_block_causal_ops",
    "lmdm_enc_dec_ops",
    "lmm_decode_ops",
    "pipeline_demo",
    "relativistic_loss",
    "routing_mask",
    "table_sketch_musdb",
    "table_text_global",
]
