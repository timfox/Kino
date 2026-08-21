"""Key citation anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, Any]]:
    return [
        {"id": 5, "cite": "Hu et al. 2026", "topic": "Demystifying NCCL (companion analysis)"},
        {"id": 9, "cite": "NVIDIA NVSHMEM docs", "topic": "OpenSHMEM PGAS for GPUs"},
        {"id": 11, "cite": "Hamidouche et al. 2025", "topic": "NCCL GPU-initiated networking (GIN)"},
        {"id": 12, "cite": "Markthub et al. 2022", "topic": "IBGDA / GPUDirect Async"},
        {"id": 27, "cite": "Langer et al. 2022", "topic": "Dynamic symmetric heap allocation"},
        {"id": 33, "cite": "Zhao et al. 2025", "topic": "DeepEP expert-parallel communication"},
    ]
