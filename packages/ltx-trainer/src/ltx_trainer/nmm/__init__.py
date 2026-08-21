"""NMM Roadmap: native multimodal modeling survey (An et al., arXiv:2605.25343)."""

from ltx_trainer.nmm.config import NMMConfig
from ltx_trainer.nmm.nativity import FusionRegime, describe_fusion, fusion_formulas
from ltx_trainer.nmm.pipeline import (
    classify_model,
    evaluation_demo,
    framework_card,
    roadmap_demo,
    table_evaluation_benchmarks,
    table_native_models,
    table_training_data,
)
from ltx_trainer.nmm.taxonomy import IOCategory, describe_io, io_formulas

__all__ = [
    "FusionRegime",
    "IOCategory",
    "NMMConfig",
    "classify_model",
    "describe_fusion",
    "describe_io",
    "evaluation_demo",
    "framework_card",
    "fusion_formulas",
    "io_formulas",
    "roadmap_demo",
    "table_evaluation_benchmarks",
    "table_native_models",
    "table_training_data",
]
