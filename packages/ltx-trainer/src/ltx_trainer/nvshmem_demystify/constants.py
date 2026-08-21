"""Paper anchors — Demystifying NVSHMEM (Ma et al., arXiv:2606.05951)."""

from __future__ import annotations

PAPER_ARXIV = "2606.05951"
PAPER_TITLE = (
    "Demystifying NVSHMEM: A System-Level Analysis on Symmetric Memory "
    "and Device-Initiated Operations in GPU Communication"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
NVSHMEM_VERSION = "3.3.9"
NVSHMEM_DOCS = "https://docs.nvidia.com/nvshmem/api/gen/index.html"

HARDWARE_MICROBENCH = "CoreWeave H200 cluster (8× H200 SXM5, NVLink-4, ConnectX-7 IB)"
NVLINK_REF_GBPS = 450.0
IB_SINGLE_RAIL_GBPS = 50.0
IB_NODE_AGG_GBPS = 400.0  # 8× 50 GB/s NICs per node

# Table I — API groups (availability summary).
API_GROUPS: tuple[dict[str, str], ...] = (
    {"group": "setup_exit", "availability": "mixed", "description": "init, finalize, bootstrap, global_exit"},
    {"group": "memory_management", "availability": "host", "description": "symmetric heap malloc/align/free, buffer registration"},
    {"group": "team_management", "availability": "mixed", "description": "teams, PE translation; create/destroy host-side"},
    {"group": "one_sided_rma", "availability": "both", "description": "put/get, iput/iget, nonblocking variants"},
    {"group": "atomics", "availability": "both", "description": "remote fetch/add, CAS, bitwise atomics on symmetric data"},
    {"group": "memory_ordering", "availability": "both", "description": "fence, quiet, quiet_on_stream"},
    {"group": "synchronization", "availability": "mixed", "description": "wait, test, signal-based waiting"},
    {"group": "collectives", "availability": "both", "description": "Barrier, Sync, Broadcast, AlltoAll, FCollect, Reduce, ReduceScatter"},
)

# Table II — collective algorithm summary (abbreviated).
TABLE2_COLLECTIVES: tuple[dict[str, str | bool], ...] = (
    {"collective": "Broadcast", "algorithm": "Bruteforce put-to-all", "ll": False, "volume": "O(M N)", "latency": "O(1)"},
    {"collective": "Broadcast", "algorithm": "k-ary flat tree", "ll": True, "volume": "O(M N)", "latency": "O(logk N)"},
    {"collective": "AlltoAll", "algorithm": "P2P / general all-push", "ll": False, "volume": "O(M N²)", "latency": "O(1)"},
    {"collective": "FCollect", "algorithm": "NVLS one-shot", "ll": True, "volume": "O(M N²)", "latency": "O(1)"},
    {"collective": "Reduce", "algorithm": "NVLS two-shot", "ll": False, "volume": "O(M N)", "latency": "O(1)"},
    {"collective": "Reduce", "algorithm": "k-ary recursive exchange", "ll": False, "volume": "O(M N k logk N)", "latency": "O(logk N)"},
    {"collective": "ReduceScatter", "algorithm": "NVLS one-shot", "ll": False, "volume": "O(M N²)", "latency": "O(1)"},
)

# Figure 4 — device-side RMA peaks (GB/s unless noted).
FIG4_RMA_PEAKS: dict[str, dict[str, float]] = {
    "intra_node": {"put_bulk": 313.2, "get_bulk": 141.0, "p_scalar": 172.0, "g_scalar": 9.0},
    "inter_node_ibgda": {"put_bulk": 48.0, "get_bulk": 48.2, "p_scalar": 15.6, "g_scalar": 1.28},
}

# Figure 4 — latency anchors at 256 B (μs).
FIG4_RMA_LATENCY_US: dict[str, dict[str, float]] = {
    "intra_node": {"put_bulk": 2.5, "get_bulk": 2.5, "p_scalar": 1.3, "g_scalar": 2.2},
    "inter_node_ibgda": {"put_bulk": 9.4, "get_bulk": 9.5, "p_scalar": 7.5, "g_scalar": 25.3},
}

# Figure 5 — AllReduce float sum peaks (algorithm bandwidth GB/s).
FIG5_ALLREDUCE_PEAKS: dict[str, float] = {
    "nvshmem_on_stream_intra": 274.8,
    "nvshmem_device_block_intra": 30.0,
    "nccl_ring_intra": 264.0,  # approximate from "outperforming ring"
    "nccl_nvls_intra": 276.0,
    "nvshmem_inter": 0.20,
    "nccl_ring_inter": 180.0,
    "nccl_nvls_tree_inter": 252.0,
}

DEEPEP_VERSION = "V1"
DEEPEP_HT_TEAMS = 8  # parallel NVSHMEM world teams per GPU slot
