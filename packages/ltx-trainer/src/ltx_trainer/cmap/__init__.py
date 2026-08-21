"""CMAP: Cross-Modal Adaptive Prompting for multi-domain task-incremental CLIP-style models (Mandalika, arXiv:2605.25708).

Reference utilities only: text-space task routing, MPVTC confidence + task-adaptive thresholds, symmetric
Hard Gumbel text gates. Frozen CLIP encoders and IAP-style prompt pools are **external** — pass tensors in.
"""

from ltx_trainer.cmap.config import CMAPConfig
from ltx_trainer.cmap.gating import (
    SymmetricTextGates,
    batch_image_mean,
    hard_gumbel_gate_values,
    symmetric_text_gate_param_count,
    text_layer_gates_from_vbar,
)
from ltx_trainer.cmap.kmeans import kmeans_torch
from ltx_trainer.cmap.mpvtc import (
    calibrate_task_thresholds,
    class_textual_confidence,
    class_visual_confidence,
    fit_visual_prototypes_per_class,
    joint_class_confidence,
    prompting_weight,
    task_confidence_from_joint,
)
from ltx_trainer.cmap.pipeline import cmap_calibrate_thresholds_from_training, cmap_infer_batch_routed
from ltx_trainer.cmap.routing import cosine_sim, route_task, task_text_prototypes

__all__ = [
    "CMAPConfig",
    "SymmetricTextGates",
    "batch_image_mean",
    "calibrate_task_thresholds",
    "cmap_calibrate_thresholds_from_training",
    "cmap_infer_batch_routed",
    "class_textual_confidence",
    "class_visual_confidence",
    "cosine_sim",
    "fit_visual_prototypes_per_class",
    "hard_gumbel_gate_values",
    "joint_class_confidence",
    "kmeans_torch",
    "prompting_weight",
    "route_task",
    "symmetric_text_gate_param_count",
    "task_confidence_from_joint",
    "task_text_prototypes",
    "text_layer_gates_from_vbar",
]
