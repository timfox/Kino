"""Reference anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, str]]:
    return [
        {"id": 1, "citation": "MPI 4.1", "topic": "Collective communication patterns"},
        {"id": 8, "citation": "Cai et al. 2021 SCCL", "topic": "SMT optimal collective synthesis"},
        {"id": 31, "citation": "Liu et al. 2024 TE-CCL", "topic": "Multi-commodity flow All-to-All"},
        {"id": 48, "citation": "Shah et al. 2023 TACCL", "topic": "ILP communication sketches"},
        {"id": 55, "citation": "Won et al. 2024 TACOS", "topic": "TEN greedy synthesis"},
        {"id": 13, "citation": "Cowan et al. 2023 MSCCLang", "topic": "GPU collective DSL / IR export"},
        {"id": 44, "citation": "Rashidi et al. 2020 ASTRA-sim", "topic": "Distributed DL simulator"},
        {"id": 29, "citation": "Li et al. 2020 PyTorch distributed", "topic": "Process groups"},
    ]
