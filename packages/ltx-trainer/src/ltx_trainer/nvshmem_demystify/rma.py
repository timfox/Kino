"""One-sided RMA fast/slow paths (§V)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvshmem_demystify.constants import FIG4_RMA_LATENCY_US, FIG4_RMA_PEAKS


def rma_path_card(peer_p2p_reachable: bool, ibgda_available: bool) -> dict[str, Any]:
    if peer_p2p_reachable:
        return {
            "path": "fast",
            "mechanism": "direct GPU load/store on mapped peer VA",
            "device": "nvshmemi_p / nvshmemi_g, threadgroup memcpy",
            "host": "cudaMemcpyAsync on mapped addresses",
        }
    if ibgda_available:
        return {
            "path": "slow_ibgda",
            "mechanism": "GPU posts RDMA WQEs via nvshmemi_ibgda_rma_*",
            "device": "kernel-initiated, no CPU proxy",
            "host": "remote transport RMA descriptors",
        }
    return {
        "path": "slow_proxy",
        "mechanism": "GPU writes proxy descriptor; host thread executes",
        "device": "nvshmemi_transfer_rma_* → proxy buffer",
        "host": "nvshmemi_proxy_rma_*",
    }


def rma_performance_summary() -> dict[str, Any]:
    return {
        "figure": "Fig. 4",
        "peaks_gbps": FIG4_RMA_PEAKS,
        "latency_us_256b": FIG4_RMA_LATENCY_US,
        "takeaway": "bulk/aggregated writes favor bandwidth; scalar g is latency-limited",
        "tuned_ibgda": "NVSHMEM_IBGDA_NUM_RC_PER_PE=64 for inter-node scalar p",
    }


def transport_backends() -> list[str]:
    return ["IBRC", "IBDEVX", "IBGDA", "UCX", "libfabric (cxi on Slingshot)"]
