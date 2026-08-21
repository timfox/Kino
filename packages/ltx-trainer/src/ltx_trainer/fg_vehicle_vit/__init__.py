"""Two-stage RT-DETR + ViT fine-grained vehicle classification (arXiv:2606.05149)."""

from ltx_trainer.fg_vehicle_vit.abstention import UNKNOWN_LABEL, classify_with_abstention
from ltx_trainer.fg_vehicle_vit.config import FgVehicleVitConfig
from ltx_trainer.fg_vehicle_vit.inference import PipelineOutput, pipeline_smoke, run_two_stage_on_detection
from ltx_trainer.fg_vehicle_vit.ltx_plan import ltx_integration_plan
from ltx_trainer.fg_vehicle_vit.metrics import accuracy, confusion_counts, per_class_prf
from ltx_trainer.fg_vehicle_vit.pipeline import (
    evaluation_demo,
    framework_card,
    table_abstention_rates,
    table_in_distribution_metrics,
    table_out_of_distribution_metrics,
    table_training_composition,
)
from ltx_trainer.fg_vehicle_vit.stage1 import Detection, filter_detections
from ltx_trainer.fg_vehicle_vit.stage2 import focal_loss, inverse_frequency_class_weights, training_step_demo
from ltx_trainer.fg_vehicle_vit.taxonomy import normalize_label, route_stage2

__all__ = [
    "Detection",
    "FgVehicleVitConfig",
    "PipelineOutput",
    "UNKNOWN_LABEL",
    "accuracy",
    "classify_with_abstention",
    "confusion_counts",
    "evaluation_demo",
    "filter_detections",
    "focal_loss",
    "framework_card",
    "inverse_frequency_class_weights",
    "ltx_integration_plan",
    "normalize_label",
    "per_class_prf",
    "pipeline_smoke",
    "route_stage2",
    "run_two_stage_on_detection",
    "table_abstention_rates",
    "table_in_distribution_metrics",
    "table_out_of_distribution_metrics",
    "table_training_composition",
    "training_step_demo",
]
