"""Reference anchors from the twelve tips paper."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, str]]:
    return [
        {"id": 1, "citation": "Alnasir 2021", "topic": "Fifteen quick tips for HPC (PLoS Comp Bio)"},
        {"id": 2, "citation": "Moreau & Wiebels 2026", "topic": "Nine quick tips for containerization"},
        {"id": 3, "citation": "Ferreira da Silva et al. 2024", "topic": "Scientific workflows + HPC integration"},
        {"id": 4, "citation": "Ejarque et al. 2022", "topic": "Dynamic intelligent workflows for HPC/AI"},
        {"id": 5, "citation": "Kurtzer et al. 2017", "topic": "Singularity/Apptainer mobility of compute"},
        {"id": 6, "citation": "Zaharia et al. 2018", "topic": "MLflow experiment lifecycle"},
        {"id": 7, "citation": "Folk et al. 2011", "topic": "HDF5 technology suite"},
        {"id": 8, "citation": "Godoy et al. 2020", "topic": "ADIOS 2 high-performance I/O"},
        {"id": 9, "citation": "Di Tommaso et al. 2017", "topic": "Nextflow reproducible workflows"},
        {"id": 10, "citation": "Mölder et al. 2021", "topic": "Snakemake sustainable analysis"},
        {"id": 11, "citation": "Amstutz et al. 2016", "topic": "Common Workflow Language v1.0"},
        {"id": 12, "citation": "Babuji et al. 2019", "topic": "Parsl parallel Python"},
        {"id": 13, "citation": "Deelman et al. 2015", "topic": "Pegasus workflow management"},
        {"id": 14, "citation": "Jain et al. 2015", "topic": "FireWorks high-throughput workflows"},
        {"id": 15, "citation": "Rocklin 2015", "topic": "Dask parallel blocked algorithms"},
        {"id": 16, "citation": "Moritz et al. 2018", "topic": "Ray distributed AI applications"},
    ]
