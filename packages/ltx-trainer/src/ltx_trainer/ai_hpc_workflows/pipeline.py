"""End-to-end AI-driven HPC workflow design demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ai_hpc_workflows.config import AiHpcWorkflowConfig
from ltx_trainer.ai_hpc_workflows.data_gravity import data_gravity_assessment, io_mitigation_plan
from ltx_trainer.ai_hpc_workflows.experiment_tracking import container_spec, experiment_record
from ltx_trainer.ai_hpc_workflows.feedback import feedback_loop_spec, iteration_gate
from ltx_trainer.ai_hpc_workflows.orchestration import (
    architecture_card,
    distributed_training_advice,
    job_array_plan,
    recommend_orchestrator,
)
from ltx_trainer.ai_hpc_workflows.phases import default_phase_plan
from ltx_trainer.ai_hpc_workflows.throughput import throughput_metrics
from ltx_trainer.ai_hpc_workflows.tips import checklist_score


def run_demo(cfg: AiHpcWorkflowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AiHpcWorkflowConfig()
    flags = {
        "integrate_ai": True,
        "data_gravity": True,
        "separate_phases": True,
        "heterogeneous_resources": True,
        "workflow_parallelism": True,
        "job_arrays_distributed": True,
        "containerise": cfg.use_containers,
        "track_experiments": True,
        "feedback_loops": cfg.feedback_enabled,
        "optimise_throughput": True,
        "manage_io": True,
        "workflow_engine": True,
    }
    return {
        "architecture": architecture_card(),
        "phases": default_phase_plan(cfg),
        "data_gravity": data_gravity_assessment(
            dataset_gb=cfg.dataset_gb,
            transfers_per_epoch=3,
            compute_hours_per_epoch=2.5,
            scratch_gb=cfg.scratch_gb,
        ),
        "job_arrays": job_array_plan(task_count=cfg.hyperparameter_trials),
        "distributed_training": distributed_training_advice(
            model_params_m=340,
            gpus_available=4,
            comm_overhead_ms=22.0,
        ),
        "orchestrator": recommend_orchestrator(cfg.adaptivity),
        "feedback": feedback_loop_spec(),
        "iteration_gate": iteration_gate({"prev_loss": 0.42, "curr_loss": 0.38}),
        "throughput": throughput_metrics(
            useful_tasks_completed=cfg.hyperparameter_trials + cfg.simulation_runs,
            wall_hours=48.0,
            queue_wait_hours=6.0,
            compute_hours=36.0,
        ),
        "io_plan": io_mitigation_plan(
            small_file_count=cfg.io_small_files,
            checkpoint_count=cfg.checkpoint_count,
            concurrent_writers=3,
        ),
        "container": container_spec(image=cfg.container_image),
        "experiment": experiment_record(
            run_id="ai-hpc-demo-001",
            hyperparameters={"lr": 3e-4, "batch_size": 32, "epochs": cfg.epochs},
            metrics={"val_auc": 0.91, "train_loss": 0.21},
            dataset_version="omics-v3.2",
            container_digest="sha256:demo",
            model_path="/scratch/models/demo.pt",
        ),
        "checklist": checklist_score(flags),
    }
