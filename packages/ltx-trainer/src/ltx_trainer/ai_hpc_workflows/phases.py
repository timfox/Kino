"""Workflow phase decomposition and resource mapping (Tips 3–4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ai_hpc_workflows.config import AiHpcWorkflowConfig


def default_phase_plan(cfg: AiHpcWorkflowConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or AiHpcWorkflowConfig()
    return [
        {
            "phase": "preprocessing",
            "queue": cfg.cpu_queue,
            "resource": "cpu",
            "parallelism": "workflow",
            "notes": "CPU-bound data prep; avoid GPU queue",
        },
        {
            "phase": "training",
            "queue": cfg.gpu_queue,
            "resource": "gpu",
            "parallelism": "distributed_or_single",
            "notes": f"{cfg.epochs} epochs; checkpoint every N steps",
        },
        {
            "phase": "inference",
            "queue": cfg.gpu_queue,
            "resource": "gpu",
            "parallelism": "job_array",
            "shards": cfg.inference_shards,
        },
        {
            "phase": "simulation",
            "queue": cfg.cpu_queue,
            "resource": "cpu",
            "parallelism": "mpi_or_task_pool",
            "runs": cfg.simulation_runs,
        },
        {
            "phase": "postprocessing",
            "queue": cfg.cpu_queue,
            "resource": "cpu",
            "parallelism": "workflow",
        },
    ]


def map_task_to_resource(task: str) -> dict[str, str]:
    task_l = task.lower()
    if any(k in task_l for k in ("train", "finetune", "backprop")):
        return {"resource": "gpu", "queue_hint": "gpu"}
    if any(k in task_l for k in ("infer", "predict", "score")):
        return {"resource": "gpu", "queue_hint": "gpu-array"}
    if any(k in task_l for k in ("simulate", "md", "fem", "mpi")):
        return {"resource": "cpu", "queue_hint": "cpu-mpi"}
    if any(k in task_l for k in ("large", "foundation", "omics")):
        return {"resource": "high_memory", "queue_hint": "mem"}
    return {"resource": "cpu", "queue_hint": "cpu"}
