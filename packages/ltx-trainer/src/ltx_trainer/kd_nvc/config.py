"""KD-NVC config (arXiv:2606.04595)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KDNVCConfig:
    paper_arxiv: str = "arXiv:2606.04595"
    teacher: str = "DCVC-RT (reproduced MT)"
    anchor: str = "VTM-LDB-23.11"
    packages: tuple[str, ...] = (
        "ae_nas",
        "efd",
        "metrics",
        "simulation",
    )


@dataclass(frozen=True)
class StudentArch:
    name: str
    target_speedup_pct: float
    inter_pred: str
    decoder: str
    recon: str


KD_NVC_S = StudentArch(
    name="KD-NVC-S",
    target_speedup_pct=60.0,
    inter_pred="1/2 L",
    decoder="1/3 L",
    recon="1/2 C",
)

KD_NVC_T = StudentArch(
    name="KD-NVC-T",
    target_speedup_pct=100.0,
    inter_pred="1/2 L 1/2 F",
    decoder="1/3 L 1/2 F",
    recon="1/2 L 1/2 C",
)


@dataclass(frozen=True)
class DistillConfig:
    pool_k: int = 8
    beta_stage1: float = 1.0
    beta_stage2: float = 0.0
    lambda_rd: float = 384.0
