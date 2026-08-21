"""Figure 4–5 microbenchmark anchors (§VII)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvshmem_demystify.constants import (
    FIG4_RMA_LATENCY_US,
    FIG4_RMA_PEAKS,
    FIG5_ALLREDUCE_PEAKS,
    HARDWARE_MICROBENCH,
    IB_NODE_AGG_GBPS,
    IB_SINGLE_RAIL_GBPS,
    NVLINK_REF_GBPS,
)


def figure4_rma() -> dict[str, Any]:
    return {
        "hardware": HARDWARE_MICROBENCH,
        "nvshmem_version": "3.3.9",
        "setup": "32 CTAs × 256 threads; CUDA Graphs for collectives",
        "references_gbps": {"nvlink": NVLINK_REF_GBPS, "ib_single_rail": IB_SINGLE_RAIL_GBPS},
        "peaks": FIG4_RMA_PEAKS,
        "latency_us": FIG4_RMA_LATENCY_US,
    }


def figure5_allreduce() -> dict[str, Any]:
    return {
        "operation": "float sum AllReduce",
        "intra_node_gpus": 8,
        "inter_node_gpus": 16,
        "references_gbps": {"nvlink_per_gpu": NVLINK_REF_GBPS, "ib_node_agg": IB_NODE_AGG_GBPS},
        "peaks": FIG5_ALLREDUCE_PEAKS,
        "takeaway": "multi-CTA on-stream + NVLS critical; inter-node NVSHMEM << NCCL",
    }


def summary_anchors() -> dict[str, Any]:
    intra = FIG4_RMA_PEAKS["intra_node"]
    ar = FIG5_ALLREDUCE_PEAKS
    return {
        "hardware": HARDWARE_MICROBENCH,
        "nvshmem_version": "3.3.9",
        "intra_put_peak_gbps": intra["put_bulk"],
        "inter_ibgda_put_peak_gbps": FIG4_RMA_PEAKS["inter_node_ibgda"]["put_bulk"],
        "allreduce_on_stream_intra_gbps": ar["nvshmem_on_stream_intra"],
        "allreduce_nccl_nvls_intra_gbps": ar["nccl_nvls_intra"],
        "allreduce_device_block_intra_gbps": ar["nvshmem_device_block_intra"],
        "role_vs_nccl": "complementary PGAS substrate; NCCL leads bulk collectives inter-node",
    }


def positioning_vs_nccl() -> dict[str, Any]:
    return {
        "nvshmem": "flat symmetric heap; device-initiated one-sided RMA/atomics",
        "nccl": "collective-centric; device API adds GIN/symmetric memory (builds on NVSHMEM ideas)",
        "nvshmem_strength": "fine-grained GPU-driven communication, irregular/sparse patterns",
        "nccl_strength": "optimized bulk-synchronous AllReduce, inter-node scale-out",
        "deepep": "custom kernels on NVSHMEM IBGDA substrate, not generic collectives",
    }
