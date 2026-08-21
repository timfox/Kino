"""VCap witness-adjudicator reward configuration (arXiv:2605.28023)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VCapConfig:
    arxiv: str = "2605.28023"
    website: str = "https://arxiv.org/abs/2605.28023"
    backbone_policy: str = "Qwen3-VL-8B-Instruct"
    reward_model: str = "frozen MLLM judge (same family in paper)"
    wcorr: float = 0.05
    wcomp: float = 0.04
    wtxt: float = 0.01
    wlocal_video: float = 0.1
    grpo_group_size: int = 8
    grpo_kl_beta: float = 1e-3
    latent_facts_N: int = 100
    score_min: int = 0
    score_max: int = 10
