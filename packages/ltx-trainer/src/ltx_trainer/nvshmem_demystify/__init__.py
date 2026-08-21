"""Demystifying NVSHMEM system study (Ma et al., arXiv:2606.05951)."""

from ltx_trainer.nvshmem_demystify.config import NvshmemDemystifyConfig
from ltx_trainer.nvshmem_demystify.mock import evaluation_smoke
from ltx_trainer.nvshmem_demystify.paper import knowledge_bundle, paper_card
from ltx_trainer.nvshmem_demystify.pipeline import run_demo

__all__ = [
    "NvshmemDemystifyConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
