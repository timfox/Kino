"""DeepEP case study — HT and LL expert-parallel paths (§VIII)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvshmem_demystify.constants import DEEPEP_HT_TEAMS, DEEPEP_VERSION


def deepep_overview() -> dict[str, Any]:
    return {
        "library": "DeepEP (DeepSeek MoE expert parallelism)",
        "version_scope": DEEPEP_VERSION,
        "substrate": "NVSHMEM for cross-node IBGDA RMA (V2 uses NCCL GIN — out of scope)",
        "workload": "sparse data-dependent all-to-all dispatch/combine",
        "gin_comparison": "NCCL GIN within ~1–2% of NVSHMEM on DeepEP kernels",
    }


def ht_path_card() -> dict[str, Any]:
    return {
        "target": "training — bandwidth over per-token latency",
        "pipeline": "cross-node RDMA between same GPU slot → intra-node NVLink fanout",
        "teams": f"{DEEPEP_HT_TEAMS} parallel NVSHMEM world teams (one PE per node per slot)",
        "assumption": "8 P2P-accessible GPUs per node",
        "phases": ["notify_dispatch (metadata)", "dispatch/combine (payloads)"],
        "nvshmem_calls": [
            "nvshmemi_ibgda_put_nbi_warp (chunked RDMA)",
            "nvshmemi_ibgda_amo_nonfetch_add (remote tail)",
        ],
        "warp_roles": "7 senders + 1 sender coord + 8 NVL receivers per channel SM",
    }


def ll_path_card() -> dict[str, Any]:
    return {
        "target": "inference — low batch, layer latency",
        "pipeline": "RDMA inter-node; no NVLink forwarding stage",
        "teams": "single global world + strided team for same-slot GPUs",
        "structure": "one grid; SMs partitioned by expert; warp groups per expert",
        "nvshmem_calls": [
            "nvshmemi_ibgda_put_nbi_warp (inter-node payload)",
            "atomic count publish to destination",
        ],
        "intra_node": "direct P2P copy into mapped receive buffer",
    }
