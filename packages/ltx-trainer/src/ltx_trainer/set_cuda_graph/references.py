"""Key citation anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, Any]]:
    return [
        {"id": 4, "cite": "NVIDIA CUDA C++ Programming Guide", "topic": "CUDA Graph execution model"},
        {"id": 6, "cite": "Ekelund et al. 2025", "topic": "Kernel batching with CUDA graphs"},
        {"id": 7, "cite": "Guevara et al. 2009", "topic": "Task parallelism in CUDA scheduler (queue model)"},
        {"id": 9, "cite": "Huang et al. 2022", "topic": "Taskflow heterogeneous task graphs"},
        {"id": 16, "cite": "Zheng et al. 2023 GRAPE", "topic": "Dynamic DNN graph execution on GPUs"},
    ]
