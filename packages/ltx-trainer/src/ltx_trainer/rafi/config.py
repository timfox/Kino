"""RaFI — Ray/Work Forwarding Infrastructure (arXiv:2605.30294)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RafiConfig:
    paper_arxiv: str = "arXiv:2605.30294"
    license: str = "Apache-2.0"
    default_ray_bytes: int = 44
    mpi_backend: str = "cuda-aware"
    sort_backend: str = "cub-radix"
    example_apps: tuple[str, ...] = (
        "VoPaT",
        "rafi/Lander",
        "SchlieRaFI",
        "rafi/StreamLines",
        "rafi/NBody",
    )
    related: tuple[str, ...] = ("Ice-T", "BriX", "Barney")
    # Alps benchmark (paper Fig. 8)
    intranode_peak_gbps: float = 133.0
    internode_peak_gbps: float = 24.0
    intranode_sustained_gbps: float = 100.0
    internode_sustained_gbps: float = 20.0
    intranode_rays_per_sec: float = 2.1e9
    internode_rays_per_sec: float = 5.0e8
    # Local GOPEX dual-GPU layout (user hardware)
    local_gpus: tuple[str, ...] = ("NVIDIA RTX Pro 6000", "NVIDIA RTX Pro 4000")
