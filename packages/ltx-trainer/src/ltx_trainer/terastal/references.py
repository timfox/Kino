"""Bibliography anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, Any]]:
    return [
        {"key": "dream", "cite": "Kim et al. ASPLOS 2023", "topic": "heterogeneity-aware real-time multi-model ML"},
        {"key": "nvdla", "cite": "NVDLA WS dataflow", "topic": "weight-stationary accelerator"},
        {"key": "shidiannao", "cite": "Du et al. ISCA 2015", "topic": "output-stationary accelerator"},
        {"key": "maestro", "cite": "Kwon et al. IEEE Micro 2020", "topic": "DNN mapping cost analysis"},
        {"key": "xrbench", "cite": "Kwon et al. MLSys 2023", "topic": "XR ML benchmark suite"},
        {"key": "daris", "cite": "Babaei & Chantem DAC 2025", "topic": "GPU virtual-deadline scheduling"},
    ]
