"""Runtime configuration for SET demos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SetCudaGraphConfig:
    batch_size: int = 8
    num_workers: int = 8
    queue_capacity: int = 4
    gpu: str = "rtx_3090"
