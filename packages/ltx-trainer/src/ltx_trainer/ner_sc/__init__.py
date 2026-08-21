"""NeR-SC: neural representation for screen content video (arXiv:2605.27024)."""

from ltx_trainer.ner_sc.config import NeRSCConfig
from ltx_trainer.ner_sc.layout import LIMITATIONS
from ltx_trainer.ner_sc.metrics import bpp_from_param_count, ms_ssim, psnr_db
from ltx_trainer.ner_sc.mgf import mgf_fuse
from ltx_trainer.ner_sc.palette import (
    init_screen_palette,
    logits_nearest_palette,
    reconstruct_ll_from_palette,
    softmax_palette_weights,
)
from ltx_trainer.ner_sc.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    pipeline_demo,
    rd_curve_anchor_points,
    skip_threshold_tradeoff,
    table_i_video16_model_sizes,
    table_ii_per_sequence,
    table_iii_ablation,
    table_iv_palette_size,
)
from ltx_trainer.ner_sc.skip import decode_with_skip, normalized_embedding_l1, should_skip_decode
from ltx_trainer.ner_sc.wavelet import haar_dwt2, haar_dwt2_rgb, haar_idwt2, haar_idwt2_rgb

__all__ = [
    "LIMITATIONS",
    "NeRSCConfig",
    "benchmarks_bundle",
    "bpp_from_param_count",
    "decode_with_skip",
    "evaluation_demo",
    "framework_card",
    "haar_dwt2",
    "haar_dwt2_rgb",
    "haar_idwt2",
    "haar_idwt2_rgb",
    "init_screen_palette",
    "logits_nearest_palette",
    "mgf_fuse",
    "ms_ssim",
    "normalized_embedding_l1",
    "pipeline_demo",
    "psnr_db",
    "rd_curve_anchor_points",
    "reconstruct_ll_from_palette",
    "should_skip_decode",
    "skip_threshold_tradeoff",
    "softmax_palette_weights",
    "table_i_video16_model_sizes",
    "table_ii_per_sequence",
    "table_iii_ablation",
    "table_iv_palette_size",
]
