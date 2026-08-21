"""Configuration for Terastal demos."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.terastal.constants import DEFAULT_ACCURACY_THRESHOLD, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


@dataclass
class TerastalConfig:
    model_name: str = "VGG11"
    n_layers: int = 11
    deadline_s: float = 1.0 / 15.0  # 15 FPS from Table II
    n_accelerators: int = 3
    accuracy_threshold: float = DEFAULT_ACCURACY_THRESHOLD
    use_variants: bool = True
    use_virtual_budgets: bool = True
    scheduler: str = "terastal"  # terastal | fcfs | edf | dream
    scenario: str = "mcv_light"
    tags: list[str] = field(default_factory=lambda: ["terastal", "rt_sched", "heterogeneous"])


__all__ = ["TerastalConfig", "PAPER_ARXIV", "PAPER_TITLE", "PAPER_URL"]
