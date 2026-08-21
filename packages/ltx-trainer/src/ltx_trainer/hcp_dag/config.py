"""HCP crack-generation DAG config (Tarasevich et al., arXiv:2606.03473)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HCPGraphRules:
    """Assumptions Sec. II.A."""

    max_vertex_degree: int = 3
    min_chain_angle_deg: float = 150.0
    max_chain_angle_deg: float = 180.0
    allow_loops: bool = False
    single_component: bool = True


@dataclass
class HCPDAGConfig:
    paper_arxiv: str = "2606.03473"
    rules: HCPGraphRules = field(default_factory=HCPGraphRules)
    leaf_generation: int = 0
