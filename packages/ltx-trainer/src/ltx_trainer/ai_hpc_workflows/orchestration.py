"""Job arrays, distributed training, workflow engines (Tips 6, 12)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ai_hpc_workflows.constants import ORCHESTRATORS_DYNAMIC, ORCHESTRATORS_PIPELINE


def job_array_plan(
    *,
    task_count: int,
    array_max: int = 1000,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Slurm job array batching stub (Tip 6)."""
    batch = batch_size or min(array_max, max(1, task_count))
    n_arrays = (task_count + batch - 1) // batch
    return {
        "task_count": task_count,
        "batch_size": batch,
        "n_job_arrays": n_arrays,
        "scheduler": "slurm",
        "mechanism": "job_array",
    }


def distributed_training_advice(
    *,
    model_params_m: float,
    gpus_available: int,
    comm_overhead_ms: float,
) -> dict[str, Any]:
    use_dist = model_params_m >= 100 and gpus_available >= 2 and comm_overhead_ms < 50
    return {
        "use_distributed_training": use_dist,
        "model_params_m": model_params_m,
        "gpus_available": gpus_available,
        "comm_overhead_ms": comm_overhead_ms,
        "fallback": "single-GPU or job-array HPO if comm dominates",
    }


def recommend_orchestrator(adaptivity: str) -> dict[str, Any]:
    """Tip 12: static pipeline vs dynamic task graph."""
    level = adaptivity.lower()
    if level == "static":
        return {
            "adaptivity": level,
            "recommended": list(ORCHESTRATORS_PIPELINE),
            "primary": "snakemake",
            "rationale": "declarative DAG with containerised steps",
        }
    if level == "iterative":
        return {
            "adaptivity": level,
            "recommended": list(ORCHESTRATORS_PIPELINE) + ["fireworks"],
            "primary": "nextflow",
            "rationale": "pipeline plus explicit iteration channels",
        }
    return {
        "adaptivity": level,
        "recommended": list(ORCHESTRATORS_DYNAMIC),
        "primary": "parsl",
        "rationale": "dynamic task spawn/terminate from intermediate AI results",
    }


def architecture_card() -> dict[str, Any]:
    """Fig. 1 adaptive AI+HPC feedback loop."""
    return {
        "components": [
            "simulation_hpc",
            "ai_training",
            "ai_inference",
            "validation",
            "workflow_orchestrator",
        ],
        "edges": [
            "simulation → training data",
            "model → inference/predictions",
            "predictions → simulation parameters",
            "validation → model refinement",
            "orchestrator → control flow + resource placement",
        ],
        "loop": "adaptive feedback between predictions and computational tasks",
    }
