"""ForestHG-Trace configuration (arXiv:2605.27590)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TaskGroup = Literal["BP", "SA", "IA", "MT", "PO"]
ReasoningLevel = Literal["H0", "H1", "H2", "H3", "H4"]
HyperedgeFamily = Literal["BEH", "REH", "CEH"]
OperatorName = Literal["read", "filter", "expand", "aggregate", "compare", "audit", "answer"]


@dataclass
class ForestHGConfig:
    arxiv: str = "2605.27590"
    num_scenes: int = 100
    neon_sites: int = 20
    tiles_per_site: int = 5
    benchmark_instances: int = 2000
    default_backbone: str = "Qwen3.5-397B-A17B"
    zipf_tail_threshold: float = 3.0
    task_groups: tuple[TaskGroup, ...] = ("BP", "SA", "IA", "MT", "PO")
    reasoning_levels: tuple[ReasoningLevel, ...] = ("H0", "H1", "H2", "H3", "H4")
    hyperedge_families: tuple[HyperedgeFamily, ...] = ("BEH", "REH", "CEH")
    operators: tuple[OperatorName, ...] = (
        "read",
        "filter",
        "expand",
        "aggregate",
        "compare",
        "audit",
        "answer",
    )
    modalities: tuple[str, ...] = field(
        default_factory=lambda: (
            "RGB",
            "CHM",
            "elevation",
            "slope",
            "aspect",
            "LAI",
            "fPAR",
            "NDVI",
            "EVI",
            "ARVI",
            "PRI",
            "SAVI",
        )
    )
