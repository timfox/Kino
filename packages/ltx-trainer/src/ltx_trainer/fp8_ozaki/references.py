"""Key citation anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, Any]]:
    return [
        {"id": 31, "cite": "Ozaki, Uchino, Imamura 2025", "topic": "Ozaki Scheme II (CRT)"},
        {"id": 37, "cite": "Uchino, Ozaki, Imamura 2026", "topic": "Ozaki II FP8 quantisation"},
        {"id": 20, "cite": "Mukunoki 2025", "topic": "DGEMM via FP8 Ozaki"},
        {"id": 26, "cite": "NVIDIA cuBLAS blog 2025", "topic": "Emulated DGEMM in cuBLAS"},
        {"id": 28, "cite": "NVIDIA Rubin blog 2026", "topic": "Emulated DGEMM column"},
        {"id": 39, "cite": "Williams et al. 2009", "topic": "Roofline model"},
        {"id": 42, "cite": "HPCwire Genesis 2026", "topic": "DOE Ozaki fallback path"},
        {"id": 19, "cite": "Matsuoka Part 2 2026", "topic": "FFT Kulisch escape route"},
    ]
