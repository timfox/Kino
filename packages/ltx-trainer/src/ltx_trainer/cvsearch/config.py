"""Configuration for CVSearch cognitive visual search (ICML 2026, arXiv:2605.23655)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SearchMode(str, Enum):
    """Assess-then-Search routing outcomes (Fig. 2a, Fig. 4)."""

    DIRECT_ANSWER = "direct_answer"
    EXPERT_SEARCH = "expert_search"
    SCAN_SEARCH = "scan_search"


@dataclass
class CVSearchConfig:
    paper_arxiv: str = "arXiv:2605.23655"
    venue: str = "ICML 2026"
    code_repo: str = "ICML26-CVSearch"

    # Thresholds (Sec. 5.1).
    tau_q: float = 0.9  # information sufficiency
    tau_q_hat: float = 0.5  # minimum sufficiency bound
    tau_v: float = 0.4  # tree pruning (visual complexity)
    delta_tau: float = 0.1  # dynamic threshold decay step (Appendix A.3)

    # SGAP clustering (Sec. 4.3.1).
    k_min: int = 4
    k_max: int = 8

    # Adaptive tree depth (Sec. 5.3).
    depth_single_object: int = 2
    depth_multi_object: int = 3

    # Priority weights (Eq. 5).
    alpha_cv: float = 0.2
    beta_co: float = 0.4
    gamma_child: float = 0.4

    # Visual expert.
    visual_expert: str = "SAM 3"

    # Default backbone for paper tables.
    default_backbone: str = "Qwen2.5-VL-7B"


def adaptive_tree_depth(num_targets: int, cfg: CVSearchConfig | None = None) -> int:
    """Rule-based depth: D=2 for m=1, D=3 for m>1 (Sec. 5.3)."""
    cfg = cfg or CVSearchConfig()
    return cfg.depth_single_object if num_targets <= 1 else cfg.depth_multi_object
