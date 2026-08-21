"""Eisbach log-barrier — DiT belief-space entropy prior (Li & Li, 2026)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EisbachConfig:
    paper_id: str = "preprint:2026-05-29"
    title: str = (
        "Entropy as a Structural Prior: How a Log-Barrier on DiT Belief Space "
        "Drives Musical Diversity and Development"
    )
    framework: str = "Eisbach"
    backbone: str = "Stable Audio 3 Medium (1.4B DiT)"
    dataset: str = "MusicCaps"
    adapter: str = "DoRA-rows"
    lora_rank: int = 16
    lora_alpha: int = 16

    # Training (§7)
    train_steps: int = 1000
    batch_size: int = 4
    learning_rate: float = 5e-5
    barrier_lambda: float = 0.5
    baseline_lambda: float = 0.0
    clip_s: float = 120.0
    cfg_scale: float = 3.0
    diffusion_steps: int = 100

    # Table 2 — structural dimensions (barrier vs baseline)
    barrier_dynamic_range_db: float = 40.0
    baseline_dynamic_range_db: float = 25.0
    optimal_lambda_low: float = 0.3
    optimal_lambda_high: float = 0.7

    # Prediction sweep
    lambda_collapse: float = 0.8

    # Characters (§7)
    characters: tuple[str, ...] = (
        "Little Piglet Prince",
        "Raccoon Mathematician",
        "Professor Pallas Cat",
        "Seal Lawyer",
    )
