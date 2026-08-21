"""Configuration for PCCL synthesis demos."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.pccl.constants import PAPER_ARXIV, PAPER_TITLE, PAPER_URL


@dataclass
class PcclConfig:
    """Default synthesis parameters."""

    n_npus: int = 16
    mesh_width: int = 4
    topology: str = "2d_mesh"  # 2d_mesh | ring | hypercube
    collective: str = "all_gather"
    process_group: list[int] | None = None
    chunk_size_kib: int = 128
    chunks_per_npu: int = 1
    max_timesteps: int = 32
    heterogeneous: bool = False
    switch_buffer: int | None = None
    tags: list[str] = field(default_factory=lambda: ["collective", "mpi", "moe"])


__all__ = ["PcclConfig", "PAPER_ARXIV", "PAPER_TITLE", "PAPER_URL"]
