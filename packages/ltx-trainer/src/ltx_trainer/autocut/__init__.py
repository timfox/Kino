"""AutoCut: end-to-end ad video editing (arXiv:2603.28366)."""

from ltx_trainer.autocut.benchmark import BENCHMARK_CASES, AdCase, case_by_id
from ltx_trainer.autocut.config import AutoCutConfig, RQVAEConfig
from ltx_trainer.autocut.inference import AutoCutEditor, EditResult, run_footage_driven_edit, run_script_driven_edit
from ltx_trainer.autocut.material_db import MaterialDatabase, build_demo_database
from ltx_trainer.autocut.metrics import MetricBundle
from ltx_trainer.autocut.mock import evaluation_smoke
from ltx_trainer.autocut.pipeline import (
    benchmarks_bundle,
    editing_demo,
    evaluation_demo,
    framework_card,
    knowledge_card,
)
from ltx_trainer.autocut.rqvae import QuantizedFeatures, ResidualRQVAE, rqvae_forward
from ltx_trainer.autocut.tables import table1_main_results, table2_ablation_training
from ltx_trainer.autocut.taxonomy import EditingTask, RenderStrategy, TrainingStage
from ltx_trainer.autocut.training import full_training_pipeline

__all__ = [
    "AdCase",
    "AutoCutConfig",
    "AutoCutEditor",
    "BENCHMARK_CASES",
    "EditResult",
    "EditingTask",
    "MaterialDatabase",
    "MetricBundle",
    "QuantizedFeatures",
    "RQVAEConfig",
    "RenderStrategy",
    "ResidualRQVAE",
    "TrainingStage",
    "benchmarks_bundle",
    "build_demo_database",
    "case_by_id",
    "editing_demo",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "full_training_pipeline",
    "knowledge_card",
    "rqvae_forward",
    "run_footage_driven_edit",
    "run_script_driven_edit",
    "table1_main_results",
    "table2_ablation_training",
]
