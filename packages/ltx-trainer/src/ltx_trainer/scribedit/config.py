"""Configuration for scribble-guided editing (Xu et al., arXiv:2605.25568)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EditTask = Literal["AD", "RM", "RP", "TR"]


@dataclass
class ScribEditConfig:
    """Coverage-then-Realism curriculum + mosaic + edit-focused loss defaults."""

    # Data scale (Sec. 4.2, Appendix B)
    stage1_synthetic_samples: int = 180_000
    stage2_real_samples: int = 4_500

    # Training (Sec. 5.1, Appendix B)
    lora_rank: int = 32
    learning_rate: float = 1e-4
    stage1_steps: int = 4_000
    stage2_steps: int = 500
    global_batch_size: int = 64

    # Edit-focused loss (Eq. 3, Sec. 4.3)
    edit_loss_lambda: float = 0.1
    edit_mask_threshold: float = 0.05

    # Multi-task mosaicking (Sec. 4.2)
    single_to_multi_ratio: tuple[int, int] = (4, 1)
    mosaic_grid_sizes: tuple[int, ...] = (2, 4)
    mosaic_layouts: tuple[str, ...] = ("1x2", "2x1", "2x2")

    # Visual distractor ablation (Sec. 3.3)
    distractor_scribble_count: tuple[int, int] = (1, 3)

    # VIBE eval strengths (Sec. 5.2) — task codes
    edit_tasks: tuple[EditTask, ...] = ("AD", "RM", "RP", "TR")

    # Cross-task baseline scores from Table 1 (for comparison helpers)
    vibe_baseline_single_task: dict[str, float] = field(
        default_factory=lambda: {
            "AD": 86.73,
            "RM": 61.15,
            "RP": 74.73,
            "TR": 30.42,
        }
    )
