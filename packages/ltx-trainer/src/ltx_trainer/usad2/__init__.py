"""USAD 2.0 — universal audio distillation (arXiv:2606.06444)."""

from ltx_trainer.usad2.config import Usad2Config
from ltx_trainer.usad2.distillation import (
    distillation_demo,
    domain_aware_weights,
    layerwise_distill_stub,
    usad_loss,
)
from ltx_trainer.usad2.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.usad2.fold import annotate_audio_save_data
from ltx_trainer.usad2.mock import evaluation_smoke
from ltx_trainer.usad2.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_benchmark_averages,
    table2_ablation_small,
    table4_inference_efficiency,
)

__all__ = [
    "Usad2Config",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "distillation_demo",
    "domain_aware_weights",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "layerwise_distill_stub",
    "pipeline_demo",
    "pipeline_demo_export",
    "table1_benchmark_averages",
    "table2_ablation_small",
    "table4_inference_efficiency",
    "usad_loss",
]
