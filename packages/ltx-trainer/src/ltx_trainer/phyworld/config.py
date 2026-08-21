"""PhyWorld: physics-faithful video world model (arXiv:2605.19242)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhyWorldConfig:
    paper_arxiv: str = "arXiv:2605.19242"
    hub_model: str = "NU-World-Model-Embodied-AI/phyworld"
    base_model: str = "Wan2.2-I2V-A14B"
    training_data: str = "OpenVid-1M"
    v2v_cond_frames: int = 17
    v2v_gt_frames: int = 49
    flow_lr: float = 1e-6
    dpo_lr: float = 1e-5
    dpo_beta: float = 100.0
    dpo_lora_rank: int = 16
    dpo_lora_alpha: int = 16
    dpo_train_pairs: int = 1000
    dpo_timestep_min: int = 901
    dpo_timestep_max: int = 999
    dpo_epochs: int = 2
    judge_model: str = "Qwen3.5-9B (fine-tuned VLM judge)"
    benchmark_prompts: int = 250
    vbench_prompts: int = 500
