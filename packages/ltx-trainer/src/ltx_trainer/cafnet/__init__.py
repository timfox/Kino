"""CAFNet — cross-attentive half-truth audio deepfake detection (arXiv:2605.29531)."""

from ltx_trainer.cafnet.config import CafNetConfig, DeepfakeClass
from ltx_trainer.cafnet.features import extract_chroma, extract_feature_triplet, extract_lfcc, extract_mfcc, feature_shapes
from ltx_trainer.cafnet.layout import LIMITATIONS
from ltx_trainer.cafnet.losses import boundary_mse, cafnet_loss, class_index, weighted_cross_entropy
from ltx_trainer.cafnet.model import cafnet_forward, mfaan_forward
from ltx_trainer.cafnet.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_iii_binary_t2,
    table_iv_unified,
    table_v_per_class,
    table_vi_localisation,
    table_viii_ablation,
    table_x_cross_dataset,
    table_xi_finetune_collapse,
)

__all__ = [
    "CafNetConfig",
    "DeepfakeClass",
    "LIMITATIONS",
    "benchmarks_bundle",
    "boundary_mse",
    "cafnet_forward",
    "cafnet_loss",
    "class_index",
    "evaluation_demo",
    "extract_chroma",
    "extract_feature_triplet",
    "extract_lfcc",
    "extract_mfcc",
    "feature_shapes",
    "framework_card",
    "headline_results",
    "mfaan_forward",
    "pipeline_demo",
    "table_iii_binary_t2",
    "table_iv_unified",
    "table_v_per_class",
    "table_vi_localisation",
    "table_viii_ablation",
    "table_x_cross_dataset",
    "table_xi_finetune_collapse",
    "weighted_cross_entropy",
]
