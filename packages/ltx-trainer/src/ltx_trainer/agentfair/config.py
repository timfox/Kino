"""AgentFAIR configuration — Chen & Pai, arXiv:2607.15781."""

from __future__ import annotations

from dataclasses import dataclass, field

PAPER_ARXIV = "2607.15781"
PAPER_TITLE = (
    "AgentFAIR: A Multi-Agent Collaborative Framework for "
    "FAIRness Evaluation of Geospatial Datasets"
)
PAPER_AUTHORS = "Ming Chen, Pranav Pai (The University of Melbourne)"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_REPO = "https://github.com/MingCHEN-Github/AgentFAIR"
PAPER_ZENODO = "https://doi.org/10.5281/zenodo.18529560"
BENCHMARK = "50 geospatial datasets × 10 repositories (diagnostic study)"

# FAIR Guiding Principles sub-ids used by AgentFAIR
SUB_PRINCIPLES: tuple[str, ...] = (
    "F1",
    "F2",
    "F3",
    "F4",
    "A1.1",
    "A1.2",
    "A2",
    "I1",
    "I2",
    "I3",
    "R1.1",
    "R1.2",
    "R1.3",
)

DIMENSION_MEMBERS: dict[str, tuple[str, ...]] = {
    "F": ("F1", "F2", "F3", "F4"),
    "A": ("A1.1", "A1.2", "A2"),
    "I": ("I1", "I2", "I3"),
    "R": ("R1.1", "R1.2", "R1.3"),
}


@dataclass
class AgentFAIRConfig:
    """Runtime knobs for the CPU stub (paper defaults where applicable)."""

    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    confidence_threshold: float = 0.5
    i1_confidence_threshold: float = 0.4
    max_retries: int = 1
    enable_critic: bool = True
    mean_cost_usd: float = 0.054
    n_datasets: int = 50
    n_repositories: int = 10
    geo_indicators: tuple[str, ...] = field(
        default_factory=lambda: (
            "EPSG",
            "ISO19115",
            "OGC",
            "WMS",
            "WFS",
            "STAC",
            "GeoSPARQL",
            "CF-NetCDF",
        )
    )
