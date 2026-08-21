"""Configuration for AI-driven HPC workflow design demos."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.ai_hpc_workflows.constants import PAPER_ARXIV, PAPER_TITLE, PAPER_URL


@dataclass
class AiHpcWorkflowConfig:
    """Default workflow design parameters for stub demos."""

    dataset_gb: float = 500.0
    epochs: int = 50
    inference_shards: int = 256
    simulation_runs: int = 32
    gpu_queue: str = "gpu"
    cpu_queue: str = "cpu"
    scratch_gb: float = 200.0
    checkpoint_count: int = 20
    hyperparameter_trials: int = 48
    use_containers: bool = True
    container_image: str = "docker://pytorch/pytorch:2.2-cuda12.1"
    experiment_backend: str = "mlflow"
    adaptivity: str = "dynamic"  # static | iterative | dynamic
    orchestrator: str = "auto"
    feedback_enabled: bool = True
    io_small_files: int = 10_000
    tags: list[str] = field(default_factory=lambda: ["computational_biology", "hpc", "ai"])


__all__ = ["AiHpcWorkflowConfig", "PAPER_ARXIV", "PAPER_TITLE", "PAPER_URL"]
