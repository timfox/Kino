"""End-to-end demo: memory + RMA + collectives + DeepEP."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvshmem_demystify.benchmarks import figure4_rma, figure5_allreduce, summary_anchors
from ltx_trainer.nvshmem_demystify.collectives import barrier_card, ll_protocol_card, multi_cta_card
from ltx_trainer.nvshmem_demystify.config import NvshmemDemystifyConfig
from ltx_trainer.nvshmem_demystify.constants import API_GROUPS, FIG5_ALLREDUCE_PEAKS
from ltx_trainer.nvshmem_demystify.deepep import deepep_overview, ht_path_card, ll_path_card
from ltx_trainer.nvshmem_demystify.memory import heap_growth_steps, remote_address_p2p, symmetric_heap_card
from ltx_trainer.nvshmem_demystify.rma import rma_path_card, transport_backends


def run_demo(cfg: NvshmemDemystifyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NvshmemDemystifyConfig()
    heap_base = 0x1_0000_0000
    peer_base = 0x2_0000_0000
    dest_local = heap_base + 4096
    dest_remote = remote_address_p2p(dest_local, heap_base, peer_base)
    f5 = figure5_allreduce()
    on_stream = FIG5_ALLREDUCE_PEAKS["nvshmem_on_stream_intra"]
    nccl_nvls = FIG5_ALLREDUCE_PEAKS["nccl_nvls_intra"]
    return {
        "config": {
            "nvshmem_version": cfg.nvshmem_version,
            "pe_count": cfg.pe_count,
            "ibgda_tuned": cfg.ibgda_tuned,
        },
        "api_group_count": len(API_GROUPS),
        "symmetric_heap": symmetric_heap_card(),
        "heap_growth": heap_growth_steps(),
        "remote_address_example": {
            "dest_local": dest_local,
            "dest_remote": dest_remote,
            "offset_preserved": dest_remote - peer_base == dest_local - heap_base,
        },
        "rma_paths": {
            "p2p": rma_path_card(True, False),
            "ibgda": rma_path_card(False, True),
            "proxy": rma_path_card(False, False),
        },
        "transports": transport_backends(),
        "collectives": {
            "ll_protocol": ll_protocol_card(),
            "multi_cta": multi_cta_card(),
            "barrier": barrier_card(),
        },
        "figure_4": figure4_rma(),
        "figure_5": f5,
        "on_stream_near_nccl_nvls": abs(on_stream - nccl_nvls) / nccl_nvls < 0.05,
        "deepep": (
            {
                "overview": deepep_overview(),
                "ht": ht_path_card(),
                "ll": ll_path_card(),
            }
            if cfg.include_deepep
            else None
        ),
        "summary": summary_anchors(),
    }
