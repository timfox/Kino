"""Key citation anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, Any]]:
    return [
        {"id": 38, "cite": "Watt ISSAC 2023", "topic": "Whole shifted inverse division"},
        {"id": 32, "cite": "Oancea & Watt 2026", "topic": "Midsize GPU add/mul (Futhark + CUDA)"},
        {"id": 29, "cite": "NVlabs CGBN 2018", "topic": "Warp-level big integers"},
        {"id": 27, "cite": "Norman & Watt CASC 2024", "topic": "Clipped products (future work)"},
        {"id": 34, "cite": "Raahauge MSc 2025", "topic": "GPU division thesis refinements"},
        {"id": 3, "cite": "Blelloch 1987", "topic": "Parallel prefix scans"},
    ]
