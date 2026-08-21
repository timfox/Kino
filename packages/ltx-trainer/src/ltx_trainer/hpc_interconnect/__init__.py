"""HPC interconnect congestion characterization (Piarulli et al., arXiv:2604.11432)."""

from ltx_trainer.hpc_interconnect.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.hpc_interconnect.experiment import run_congestion_experiment
from ltx_trainer.hpc_interconnect.fabrics import AggressorPattern, SystemName, get_system
from ltx_trainer.hpc_interconnect.paper import evaluation_demo, framework_card

__all__ = [
    "AggressorPattern",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "SystemName",
    "benchmarks_bundle",
    "evaluation_demo",
    "framework_card",
    "get_system",
    "run_congestion_experiment",
]
