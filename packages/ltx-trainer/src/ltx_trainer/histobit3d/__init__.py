"""HistoBIT3D: virtual 3D H&E from BIT (Song et al., arXiv:2605.22000)."""

from ltx_trainer.histobit3d.config import BASELINES, DATASET_SUBSETS, HistoBIT3DConfig, MSC_SCALES
from ltx_trainer.histobit3d.losses import (
    adain_token,
    bidirectional_msc_loss,
    multiscale_content_loss,
    style_statistics_loss,
    total_histobit_loss,
    update_style_prototype,
)
from ltx_trainer.histobit3d.metrics import dice_3d, evaluate_3d_segmentation, hausdorff95
from ltx_trainer.histobit3d.pipeline import (
    ablation_table,
    benchmark_table_sota,
    dataset_card,
    demo_segmentation_metrics,
    preprocess_bit_slice,
    training_step_demo,
)
from ltx_trainer.histobit3d.preprocessing import background_subtract, bit_three_channel_stack, scale_to_uint8

__all__ = [
    "BASELINES",
    "DATASET_SUBSETS",
    "HistoBIT3DConfig",
    "MSC_SCALES",
    "ablation_table",
    "adain_token",
    "background_subtract",
    "benchmark_table_sota",
    "bidirectional_msc_loss",
    "bit_three_channel_stack",
    "dataset_card",
    "demo_segmentation_metrics",
    "dice_3d",
    "evaluate_3d_segmentation",
    "hausdorff95",
    "multiscale_content_loss",
    "preprocess_bit_slice",
    "scale_to_uint8",
    "style_statistics_loss",
    "total_histobit_loss",
    "training_step_demo",
    "update_style_prototype",
]
