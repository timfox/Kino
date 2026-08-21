"""Reference anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, str]]:
    return [
        {"id": 1, "citation": "Berenger 1994", "topic": "PML absorbing layers"},
        {"id": 6, "citation": "Micikevicius 2009", "topic": "3D FDTD on CUDA GPUs"},
        {"id": 10, "citation": "Roden & Gedney 2000", "topic": "Convolutional PML (CPML)"},
        {"id": 2, "citation": "Kamil et al. 2006", "topic": "Stencil temporal blocking"},
        {"id": 5, "citation": "Meng & Skadron 2009", "topic": "Ghost-zone GPU optimization"},
        {"id": 9, "citation": "Pasalic & McGarry 2010", "topic": "Acoustic CPML validation"},
    ]
