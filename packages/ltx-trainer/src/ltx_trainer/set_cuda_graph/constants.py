"""Paper anchors — SET CUDA Graph scheduling (Li et al., arXiv:2606.05495)."""

from __future__ import annotations

PAPER_ARXIV = "2606.05495"
PAPER_TITLE = "SET: Stream-Event-Triggered Scheduling for Efficient CUDA Graph Pipelines"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

CUDA_VERSION = "12.8"
HARDWARE = {
    "rtx_3090": "Intel Xeon Gold 6330 + NVIDIA RTX 3090 (Ampere), 192 GB RAM",
    "rtx_5090": "Intel i7-11700 + NVIDIA RTX 5090 (Blackwell), 128 GB RAM",
}

BASELINE_MODELS: tuple[str, ...] = (
    "Synchronous",
    "Graph",
    "Batching",
    "Queue",
    "SET",
)

WORKLOADS: tuple[dict[str, str], ...] = (
    {"name": "Sobel", "metric": "images/ms", "character": "short_bw_intensive"},
    {"name": "SSSP", "metric": "tasks/s", "character": "short_bw_light"},
    {"name": "BP", "metric": "tasks/s", "character": "short_bw_light"},
    {"name": "GEMM", "metric": "GFLOPs", "character": "long_bw_intensive"},
    {"name": "KNN", "metric": "queries/ms", "character": "short_bw_light"},
    {"name": "Hotspot", "metric": "grids/s", "character": "long_bw_intensive"},
)

# Table 1 — speedups vs baselines (average Ampere + Blackwell columns from paper).
TABLE1_SPEEDUPS: tuple[dict[str, str | float], ...] = (
    {
        "workload": "Sobel",
        "sync_ampere": 2.99,
        "sync_blackwell": 1.86,
        "graph_ampere": 2.97,
        "graph_blackwell": 1.71,
        "batch_ampere": 1.23,
        "batch_blackwell": 1.12,
        "queue_ampere": 1.20,
        "queue_blackwell": 1.10,
    },
    {
        "workload": "SSSP",
        "sync_ampere": 2.45,
        "sync_blackwell": 3.07,
        "graph_ampere": 2.46,
        "graph_blackwell": 2.99,
        "batch_ampere": 1.15,
        "batch_blackwell": 1.07,
        "queue_ampere": 1.10,
        "queue_blackwell": 1.03,
    },
    {
        "workload": "BP",
        "sync_ampere": 2.34,
        "sync_blackwell": 2.78,
        "graph_ampere": 2.26,
        "graph_blackwell": 2.78,
        "batch_ampere": 1.10,
        "batch_blackwell": 1.04,
        "queue_ampere": 1.01,
        "queue_blackwell": 1.01,
    },
    {
        "workload": "GEMM",
        "sync_ampere": 1.58,
        "sync_blackwell": 1.39,
        "graph_ampere": 1.58,
        "graph_blackwell": 1.38,
        "batch_ampere": 1.12,
        "batch_blackwell": 1.02,
        "queue_ampere": 1.01,
        "queue_blackwell": 1.01,
    },
    {
        "workload": "KNN",
        "sync_ampere": 2.47,
        "sync_blackwell": 2.63,
        "graph_ampere": 2.08,
        "graph_blackwell": 2.19,
        "batch_ampere": 1.23,
        "batch_blackwell": 1.08,
        "queue_ampere": 2.94,
        "queue_blackwell": 2.09,
    },
    {
        "workload": "Hotspot",
        "sync_ampere": 1.10,
        "sync_blackwell": 1.47,
        "graph_ampere": 1.39,
        "graph_blackwell": 1.45,
        "batch_ampere": 1.27,
        "batch_blackwell": 1.56,
        "queue_ampere": 1.38,
        "queue_blackwell": 1.81,
    },
)

TABLE1_AVERAGE_SPEEDUP = {
    "Synchronous": {"ampere": 2.15, "blackwell": 2.20, "combined": 2.18},
    "Graph": {"ampere": 2.12, "blackwell": 2.08, "combined": 2.10},
    "Batching": {"ampere": 1.18, "blackwell": 1.15, "combined": 1.17},
    "Queue": {"ampere": 1.44, "blackwell": 1.34, "combined": 1.39},
}

# Table 2 — average scheduling overhead ratio t_schedule / T_measured.
TABLE2_OVERHEAD_RATIO = {
    "rtx_3090": {"Batching": 0.4532, "Queue": 0.3336, "SET": 0.2983},
    "rtx_5090": {"Batching": 0.5238, "Queue": 0.3985, "SET": 0.3462},
}

OVERHEAD_REDUCTION_VS = {
    "batching_pct": 54.64,
    "queue_pct": 18.62,
}

# Fig. 6 anchors at batch size 4096 (SET vs queue overhead narrative).
FIG6_BATCH4096_OVERHEAD_PCT = {"Batching": 59, "Queue": 54, "SET": 45}
FIG6_BATCH4_OVERHEAD_PCT = {"Batching": 25, "Queue": 18, "SET": 11}

THROUGHPUT_SPEEDUP_RANGE = (1.15, 1.44)
