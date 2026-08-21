"""Configuration for FDTD+CPML multi-GPU demos."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.fdtd_cpml_multigpu.constants import PAPER_ARXIV, PAPER_TITLE, PAPER_URL


@dataclass
class FdtdCpmlConfig:
    nx: int = 320
    ny: int = 320
    nz: int = 320
    n_steps: int = 100
    n_gpus: int = 4
    decomposition: str = "pencil_yz"
    exchange: str = "peer"  # peer | host_staged
    comm_interval: int = 1  # s ∈ {1,2,4,8}
    cpml_enabled: bool = True
    precision: str = "float32"
    tags: list[str] = field(default_factory=lambda: ["fdtd", "cpml", "cuda"])


__all__ = ["FdtdCpmlConfig", "PAPER_ARXIV", "PAPER_TITLE", "PAPER_URL"]
