"""Key citation anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, Any]]:
    return [
        {"id": 10, "cite": "Potter & Moses ASC", "topic": "Attributed scattering centers for SAR ATR"},
        {"id": 25, "cite": "Sun et al. HRNet", "topic": "High-resolution pose estimation backbone"},
        {"id": 31, "cite": "Wang et al. SAR-AIRcraft-1.0", "topic": "Gaofen-3 aircraft detection dataset"},
        {"id": 32, "cite": "Sun et al. SCAN", "topic": "Cross-dataset SAR aircraft classification"},
        {"id": 15, "cite": "Li et al. SARATR-X", "topic": "SAR foundation model baseline"},
    ]
