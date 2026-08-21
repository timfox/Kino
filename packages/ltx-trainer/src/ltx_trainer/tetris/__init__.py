"""Tetris: tile-level sampling for video object tracking (Kittivorawong et al., arXiv:2605.25538)."""

from ltx_trainer.tetris.config import TetrisConfig
from ltx_trainer.tetris.gaps import laplace_miss_rate, max_gap_matrix
from ltx_trainer.tetris.pack import ffd_pack
from ltx_trainer.tetris.pipeline import (
    evaluation_demo,
    framework_card,
    table_ablation_operators,
    table_detection_dominance,
    table_packing_efficacy,
    table_related_benchmarks,
    table_system_speedups,
    training_step_demo,
)
from ltx_trainer.tetris.prune import prune_polyominoes
from ltx_trainer.tetris.tiles import Polyomino, connected_polyominoes, frame_difference

__all__ = [
    "Polyomino",
    "TetrisConfig",
    "connected_polyominoes",
    "evaluation_demo",
    "ffd_pack",
    "framework_card",
    "frame_difference",
    "laplace_miss_rate",
    "max_gap_matrix",
    "prune_polyominoes",
    "table_ablation_operators",
    "table_detection_dominance",
    "table_packing_efficacy",
    "table_related_benchmarks",
    "table_system_speedups",
    "training_step_demo",
]
