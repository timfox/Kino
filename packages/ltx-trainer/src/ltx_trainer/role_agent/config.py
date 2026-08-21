"""Role-Agent configuration (arXiv:2606.10917)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoleAgentConfig:
    arxiv: str = "2606.10917"
    title: str = "Role-Agent: Bootstrapping LLM Agents via Dual-Role Evolution"
    github: str = "https://github.com/AMAP-ML/roleagent"
    institution: str = "AMAP, Alibaba Group"
    backbone_default: str = "Qwen2.5-1.5B-Instruct"

    # Table 5 hyper-parameters
    learning_rate: float = 1e-6
    group_size: int = 8
    gamma: float = 0.99
    kl_coefficient: float = 1e-3
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    state_similarity_threshold: float = 0.9
    advantage_alpha: float = 1.0
    rollout_temperature: float = 0.9
    reflection_temperature: float = 0.5
    total_epochs: int = 150

    # T_max per domain → H = 5% · T_max (Table 4 default)
    t_max: dict[str, int] = field(
        default_factory=lambda: {"alfworld": 50, "webshop": 15, "search_qa": 4}
    )
    prediction_horizon_frac: float = 0.05

    # Component toggles (Table 3 ablations)
    enable_wia: bool = True
    enable_aiw: bool = True

    # Search-QA retriever (paper: E5)
    search_retriever: str = "auto"  # auto | e5 | overlap

    def prediction_horizon(self, domain: str) -> int:
        t = self.t_max.get(domain, 50)
        return max(1, int(t * self.prediction_horizon_frac + 0.5))
