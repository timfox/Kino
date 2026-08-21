"""Yi — in-place graph-based vector index updates (Liu, He, Tang; arXiv:2607.15576)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2607.15576"
PAPER_TITLE = "Efficient and Effective In-place Graph-based Vector Index Updates"
PAPER_SYSTEM = "Yi"
PAPER_AUTHORS = "Haotian Liu, Yujun He, Bo Tang (SUSTech)"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
BENCHMARK = "DEEP100M / SIFT800M streaming replace workloads"

COMPONENTS = (
    "vector_level_update",
    "tasklet_engine",
    "async_buffer_manager",
    "vector_file_system",
)


@dataclass
class YiConfig:
    """Runtime knobs for the CPU stub."""

    max_out_degree: int = 96
    buffer_gb: float = 4.0
    delete_lru_frac: float = 0.04
    worker_threads: int = 12
    enable_tasklets: bool = True
    enable_buffer_persist: bool = True
    enable_split_layout: bool = True
