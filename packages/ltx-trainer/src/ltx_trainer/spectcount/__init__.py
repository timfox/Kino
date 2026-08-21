"""SpectCount — synthetic spectrotemporal counting for LALMs (arXiv:2606.06907)."""

from ltx_trainer.spectcount.config import SpectCountConfig
from ltx_trainer.spectcount.eval import eval_smoke, pipeline_demo
from ltx_trainer.spectcount.fold import annotate_audio_save_data
from ltx_trainer.spectcount.mock import evaluation_smoke
from ltx_trainer.spectcount.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_benchmarks,
    table2_signal_config,
    table3_ablation,
)
from ltx_trainer.spectcount.probing import apply_spectcount_boost, probing_demo, probing_grid
from ltx_trainer.spectcount.signals import generate_signal, signal_demo, synthesize_pulse

__all__ = [
    "SpectCountConfig",
    "annotate_audio_save_data",
    "apply_spectcount_boost",
    "benchmarks_bundle",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "generate_signal",
    "headline_results",
    "pipeline_demo",
    "probing_demo",
    "probing_grid",
    "signal_demo",
    "synthesize_pulse",
    "table1_benchmarks",
    "table2_signal_config",
    "table3_ablation",
]
