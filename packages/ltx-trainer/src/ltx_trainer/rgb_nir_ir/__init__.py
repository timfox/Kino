"""Ambient-robust inverse rendering with active RGB–NIR imaging (arXiv:2605.30250)."""

from ltx_trainer.rgb_nir_ir.config import RgbNirIrConfig
from ltx_trainer.rgb_nir_ir.mock import evaluation_smoke
from ltx_trainer.rgb_nir_ir.pipeline import benchmarks_bundle, evaluation_demo, framework_card

__all__ = [
    "RgbNirIrConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "benchmarks_bundle",
]
