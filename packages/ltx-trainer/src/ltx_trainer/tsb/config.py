"""Time Segmented Beamforming via DP (arXiv:2605.24825)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TsbConfig:
    paper_arxiv: str = "arXiv:2605.24825"
    companion_online: str = "arXiv:2605.08554"
    default_penalty_c: float = 4.8
    min_segment_length_tau: int = 5
    diagonal_loading_delta: float = 1e-3
    fir_length_l: int = 16
    ula_elements_abrupt: int = 9
    ula_elements_piecewise: int = 15
    swellex_penalty_c: float = 0.1
    swellex_min_segment_tau: int = 1
    osb_search_limit_k: int = 60
