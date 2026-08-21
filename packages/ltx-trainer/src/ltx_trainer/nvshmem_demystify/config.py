"""Runtime configuration for NVSHMEM study demos."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.nvshmem_demystify.constants import NVSHMEM_VERSION


@dataclass
class NvshmemDemystifyConfig:
    """Knobs for benchmark card reproduction."""

    nvshmem_version: str = NVSHMEM_VERSION
    pe_count: int = 8
    include_deepep: bool = True
    ibgda_tuned: bool = True
    ibgda_rc_per_pe: int = 64
