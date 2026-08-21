"""Twelve quick tips for AI-driven HPC workflows (Alnasir, arXiv:2606.07491)."""

from __future__ import annotations

PAPER_ARXIV = "2606.07491"
PAPER_TITLE = "Twelve quick tips for designing AI-driven HPC workflows"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
AUTHOR = "Dr Jamie J. Alnasir"
AFFILIATION = "Royal Holloway University of London"

KEYWORDS = (
    "High-Performance Computing (HPC)",
    "AI-driven Workflows",
    "Workflow Orchestration",
    "Distributed Model Training",
    "Containerisation",
    "Foundation Models",
    "Computational Biology",
)

WORKFLOW_PHASES = ("training", "inference", "simulation", "preprocessing", "postprocessing")

RESOURCE_TYPES = ("cpu", "gpu", "high_memory", "remote")

CONTAINER_STACKS = ("apptainer", "singularity_ce")

IO_FORMATS = ("hdf5", "adios2", "archive", "scratch_staging")

ORCHESTRATORS_PIPELINE = ("nextflow", "snakemake", "cwl")
ORCHESTRATORS_DYNAMIC = ("parsl", "pegasus", "fireworks", "dask", "ray")

TIPS: tuple[tuple[int, str, str], ...] = (
    (1, "integrate_ai", "Treat AI as part of the workflow, not a bolt-on"),
    (2, "data_gravity", "Design for data gravity, not just compute locality"),
    (3, "separate_phases", "Separate training, inference, and simulation phases"),
    (4, "heterogeneous_resources", "Use heterogeneous resources intentionally"),
    (5, "workflow_parallelism", "Exploit parallelism at the workflow level, not just the job level"),
    (6, "job_arrays_distributed", "Use job arrays and distributed training strategically"),
    (7, "containerise", "Containerise environments to ensure reproducibility"),
    (8, "track_experiments", "Track experiments as first-class outputs"),
    (9, "feedback_loops", "Design feedback loops explicitly"),
    (10, "optimise_throughput", "Optimise for throughput, not single-job performance"),
    (11, "manage_io", "Manage I/O and small files carefully"),
    (12, "workflow_engine", "Use a workflow engine to orchestrate AI + HPC pipelines"),
)
