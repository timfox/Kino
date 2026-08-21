"""MUSTBENCH — temporal grounding in music LALMs (arXiv:2605.29300)."""

from ltx_trainer.mustbench.config import MustBenchConfig, MustTask
from ltx_trainer.mustbench.eval_suite import eval_suite_smoke, run_eval_suite
from ltx_trainer.mustbench.grpo_trainer import GrpoCheckpoint, grpo_step, grpo_step_from_qa, grpo_trainer_smoke, run_grpo_epochs
from ltx_trainer.mustbench.must_encoder import encode_audio, must_encoder_smoke, transition_probability
from ltx_trainer.mustbench.layout import LIMITATIONS
from ltx_trainer.mustbench.metrics import (
    clap_score_proxy,
    hit_at_t,
    mc_accuracy,
    meteor_proxy,
    metrics_smoke,
    temporal_iou_f1,
)
from ltx_trainer.mustbench.must_train import (
    MustStage,
    bce_dice_loss,
    ccc_loss,
    dynamic_sampling_weights,
    must_train_smoke,
    sft_loss_balanced,
    stage_pipeline,
)
from ltx_trainer.mustbench.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_i_benchmark_stats,
    table_iii_main_results,
    table_iv_ablation,
)
from ltx_trainer.mustbench.rewards import grpo_smoke, mtr_grpo_reward, tsg_grpo_reward
from ltx_trainer.mustbench.tasks import split_statistics, task_registry

__all__ = [
    "GrpoCheckpoint",
    "LIMITATIONS",
    "MustBenchConfig",
    "MustStage",
    "MustTask",
    "bce_dice_loss",
    "benchmarks_bundle",
    "ccc_loss",
    "clap_score_proxy",
    "dynamic_sampling_weights",
    "encode_audio",
    "eval_suite_smoke",
    "evaluation_demo",
    "framework_card",
    "GrpoCheckpoint",
    "grpo_smoke",
    "grpo_step",
    "grpo_step_from_qa",
    "grpo_trainer_smoke",
    "headline_results",
    "hit_at_t",
    "mc_accuracy",
    "meteor_proxy",
    "metrics_smoke",
    "mtr_grpo_reward",
    "must_train_smoke",
    "must_encoder_smoke",
    "pipeline_demo",
    "run_eval_suite",
    "run_grpo_epochs",
    "sft_loss_balanced",
    "split_statistics",
    "stage_pipeline",
    "table_i_benchmark_stats",
    "table_iii_main_results",
    "table_iv_ablation",
    "task_registry",
    "temporal_iou_f1",
    "transition_probability",
    "tsg_grpo_reward",
]
