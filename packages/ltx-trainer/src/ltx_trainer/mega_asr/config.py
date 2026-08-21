"""Mega-ASR: in-the-wild robust ASR framework (arXiv:2605.19833)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MegaASRConfig:
    paper_arxiv: str = "arXiv:2605.19833"
    project_page: str = "https://xzf-thu.github.io/Mega-ASR/"
    hub_dataset: str = "zhifeixie/Voices-in-the-Wild-2M"
    bench_repo: str = "https://github.com/xzf-thu/Voices-in-the-Wild-Bench"
    backbone: str = "Qwen3-ASR-1.7B"
    dataset_clips: int = 2_400_000
    dataset_hours: float = 11_000.0
    atomic_phenomena: int = 7
    compound_scenarios: int = 54
    # DG-WGPO hyperparameters (paper Sec. 4.2)
    wer_gate_tau: float = 0.3
    soft_error_alpha: float = 0.4
    dynamic_reward_weight: float = 0.6
    rl_rollouts_k: int = 16
    rl_steps: int = 6000
    transition_wer_filter: float = 0.70
