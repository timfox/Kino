"""Key citation anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, Any]]:
    return [
        {"id": 7, "cite": "Ouyang et al. ICML 2025", "topic": "KernelBench GPU kernel generation"},
        {"id": 8, "cite": "Chen et al. arXiv:2506.09092", "topic": "CUDA-LLM efficient kernel writing"},
        {"id": 9, "cite": "Zhang et al. arXiv:2511.01884", "topic": "CudaForge agent + hardware feedback"},
        {"id": 3, "cite": "Chen et al. arXiv:2412.19770", "topic": "Fortran2CPP multi-turn LLM translation"},
        {"id": 16, "cite": "De Tomasi et al. EASE 2025", "topic": "LLM simplification bias"},
        {"id": 22, "cite": "Benjamini & Hochberg 1995", "topic": "BH-FDR multiple testing"},
    ]
