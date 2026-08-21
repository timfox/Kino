"""Experiment tracking as first-class outputs (Tip 8)."""

from __future__ import annotations

from typing import Any


def experiment_record(
    *,
    run_id: str,
    hyperparameters: dict[str, Any],
    metrics: dict[str, float],
    dataset_version: str,
    container_digest: str,
    model_path: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "provenance": {
            "dataset_version": dataset_version,
            "container_digest": container_digest,
            "software_stack": "pinned via Apptainer image",
        },
        "artefacts": {
            "model_path": model_path,
            "primary_outputs": ["model", "metrics", "config"],
        },
        "backend": "mlflow",
    }


def container_spec(
    *,
    image: str,
    runtime: str = "apptainer",
) -> dict[str, Any]:
    return {
        "runtime": runtime,
        "image": image,
        "bind_mounts": ["/scratch", "/project"],
        "notes": "Tip 7: CUDA/Python/MPI compatibility frozen in image",
    }
