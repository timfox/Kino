"""LOPEO: stimulus reconstruction AAD on unbalanced EEG datasets (IEEE SPL)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LopeoConfig:
    venue: str = "IEEE Signal Processing Letters"
    github_repo: str = "https://github.com/SeanZhang99/SuperHugeAAD"
    model: str = "VLAAI"
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    lr_factor: float = 0.5
    lr_patience_epochs: int = 5
    lr_cooldown_epochs: int = 5
    early_stop_patience: int = 10
    max_epochs: int = 100
    optimizer: str = "Adam"
