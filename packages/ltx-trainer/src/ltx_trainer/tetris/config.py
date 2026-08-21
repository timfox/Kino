"""Tetris configuration (arXiv:2605.25538)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TetrisConfig:
    """Reference hyperparameters from the Tetris paper."""

    paper_arxiv: str = "arXiv:2605.25538"
    project_page: str = "https://tetris-db.github.io"
    tile_size_px: int = 32
    relevance_threshold: float = 0.5
    gamma_candidates: tuple[int, ...] = (1, 2, 4, 8, 16)
    mistrack_tolerances: tuple[float | None, ...] = (None, 0.4, 0.6, 0.8)
    frame_sample_rates: tuple[int, ...] = (1, 2, 4, 8, 16)
    num_datasets: int = 7
    hota_loss_bound_pct: float = 5.0
    mean_irrelevant_tile_pct: float = 94.5
    roi_overhead_pct: float = 40.0
