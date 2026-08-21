"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvshmem_demystify.benchmarks import (
    figure4_rma,
    figure5_allreduce,
    positioning_vs_nccl,
    summary_anchors,
)
from ltx_trainer.nvshmem_demystify.collectives import collective_table, psync_card
from ltx_trainer.nvshmem_demystify.constants import API_GROUPS, NVSHMEM_VERSION, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.nvshmem_demystify.deepep import deepep_overview, ht_path_card, ll_path_card
from ltx_trainer.nvshmem_demystify.memory import symmetric_heap_card
from ltx_trainer.nvshmem_demystify.references import reference_anchors
from ltx_trainer.nvshmem_demystify.rma import rma_performance_summary


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "authors": "Ma, Shen, Chen, Langer, Kraus, Glick, Belusar, Hammond, Hoefler",
        "affiliations": "ETH Zürich; NVIDIA",
        "nvshmem_version": NVSHMEM_VERSION,
        "findings": [
            "Symmetric heap via CUDA VMM; offset-based remote addressing (P2P fast / transport slow)",
            "Device RMA: P2P direct access or IBGDA/proxy slow path",
            "Collectives: rule-based algorithm tree; NVLS multi-CTA on-stream for intra-node AR",
            "DeepEP uses NVSHMEM as IBGDA substrate for MoE expert parallelism",
            "Complements NCCL — fine-grained one-sided vs bulk collective centric",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "api_groups": [dict(g) for g in API_GROUPS],
        "symmetric_heap": symmetric_heap_card(),
        "collectives_table": collective_table(),
        "psync": psync_card(),
        "rma": rma_performance_summary(),
        "figure_4": figure4_rma(),
        "figure_5": figure5_allreduce(),
        "deepep": {
            "overview": deepep_overview(),
            "ht": ht_path_card(),
            "ll": ll_path_card(),
        },
        "positioning": positioning_vs_nccl(),
        "summary": summary_anchors(),
        "references": reference_anchors(),
    }
