"""GS-NFS configuration (arXiv:2606.05650)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GSNFSParams:
    octree_depth: int = 12
    rlgr_block_size: int = 512
    quant_step_opacity: float = 0.05
    quant_step_sh: float = 0.1
    ans_entropy: bool = True
    klt_decorrelation: bool = True
    target_fps_ms: float = 33.0


@dataclass(frozen=True)
class GSNFSConfig:
    paper_arxiv: str = "arXiv:2606.05650"
    bytes_per_gaussian: int = 236
    datasets: tuple[str, ...] = ("HiFi4G", "N3DV")
    baselines: tuple[str, ...] = ("V3-2D", "MesonGS", "LTS-Draco", "G-PCC")
    packages: tuple[str, ...] = ("gs_nfs",)
    params: GSNFSParams = field(default_factory=GSNFSParams)
