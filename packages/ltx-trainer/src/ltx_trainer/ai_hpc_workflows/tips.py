"""Twelve quick tips catalog and design checklist."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ai_hpc_workflows.constants import TIPS


def tips_card() -> dict[str, Any]:
    return {
        "count": len(TIPS),
        "tips": [
            {
                "id": tip_id,
                "slug": slug,
                "title": title,
            }
            for tip_id, slug, title in TIPS
        ],
    }


def tip_detail(slug: str) -> dict[str, Any] | None:
    extras: dict[str, dict[str, Any]] = {
        "integrate_ai": {
            "pitfall": "ML bolted on as isolated step",
            "practice": [
                "define train/update cadence",
                "wire predictions to downstream HPC",
                "validate and feed results back iteratively",
            ],
        },
        "data_gravity": {
            "pitfall": "repeated cross-filesystem transfers dominate runtime",
            "practice": [
                "stage data on node-local scratch",
                "minimise redundant transfers between AI and simulation",
                "heuristic: moves > compute ⇒ redesign",
            ],
        },
        "separate_phases": {
            "pitfall": "GPU queue for CPU-only preprocessing",
            "practice": [
                "GPU queues for training",
                "job arrays for inference sweeps",
                "independent CPU/MPI simulation scheduling",
            ],
        },
        "heterogeneous_resources": {
            "pitfall": "GPU requested for all stages",
            "practice": [
                "CPU for pre/post",
                "GPU for train/inference",
                "high-memory nodes for large models/datasets",
            ],
        },
        "workflow_parallelism": {
            "pitfall": "only scale single monolithic job",
            "practice": [
                "many independent tasks",
                "async where tolerable",
                "AI-guided selective execution",
            ],
        },
        "job_arrays_distributed": {
            "pitfall": "over-synchronisation in distributed training",
            "practice": [
                "Slurm job arrays for HPO/eval/inference",
                "distributed training only when comm overhead justified",
                "dynamic spawn/terminate for adaptive workflows",
            ],
        },
        "containerise": {
            "pitfall": "fragile host Python/CUDA/MPI stacks",
            "practice": [
                "Apptainer/SingularityCE images",
                "pin framework + driver compatibility",
                "portable reproducibility across clusters",
            ],
        },
        "track_experiments": {
            "pitfall": "models/metrics treated as incidental logs",
            "practice": [
                "hyperparameters + metrics as primary outputs",
                "dataset/software versions recorded",
                "MLflow or structured provenance store",
            ],
        },
        "feedback_loops": {
            "pitfall": "manual ad hoc iteration",
            "practice": [
                "encode simulation→model→simulation loops",
                "conditional execution and state tracking",
                "spawn/terminate tasks from intermediate results",
            ],
        },
        "optimise_throughput": {
            "pitfall": "micro-optimise one job while queue/I/O starve workflow",
            "practice": [
                "useful outcomes per compute hour",
                "reduce queue wait and scheduler idle time",
                "balance resources across concurrent stages",
            ],
        },
        "manage_io": {
            "pitfall": "metadata storms from checkpoints/logs",
            "practice": [
                "aggregate small files",
                "HDF5/ADIOS2 for scientific access patterns",
                "scratch for intermediates",
            ],
        },
        "workflow_engine": {
            "pitfall": "unmaintainable shell-script DAGs",
            "practice": [
                "Nextflow/Snakemake/CWL for static pipelines",
                "Parsl/Pegasus/FireWorks/Dask/Ray for dynamic graphs",
                "scheduler integration + reproducible definitions",
            ],
        },
    }
    for tip_id, s, title in TIPS:
        if s == slug:
            base = {"id": tip_id, "slug": s, "title": title}
            base.update(extras.get(s, {}))
            return base
    return None


def checklist_score(flags: dict[str, bool]) -> dict[str, Any]:
    """Score workflow design against tip slugs (True = addressed)."""
    slugs = [s for _, s, _ in TIPS]
    addressed = [s for s in slugs if flags.get(s, False)]
    missing = [s for s in slugs if not flags.get(s, False)]
    return {
        "addressed": addressed,
        "missing": missing,
        "score": round(len(addressed) / len(slugs), 3),
        "n_addressed": len(addressed),
        "n_total": len(slugs),
    }
