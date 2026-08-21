"""LatentSkill configuration (arXiv:2606.06087)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LatentSkillConfig:
    paper_arxiv: str = "2606.06087"
    github: str = "https://github.com/yuaofan0-oss/LatentSkill"
    website: str = "https://arxiv.org/abs/2606.06087"

    backbone: str = "Qwen3-8B"
    lora_rank: int = 64
    stub_lora_rank: int = 16
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05
    default_injection_alpha: float = 0.6

    # Target modules (Qwen3-8B): attn_q/k/v/o + mlp_gate/up/down; paper uses full then ablates o+d
    target_modules: tuple[str, ...] = (
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    )
    preferred_modules: tuple[str, ...] = ("attn_o", "mlp_down")
    num_layers: int = 36
    stub_layer_range: tuple[int, ...] = (34, 35)  # 2 layers for CPU stub; paper uses all 36
    stub_modules: tuple[str, ...] = ("attn_o", "mlp_down")
    stub_hidden_dim: int = 64
    hidden_dim: int = 4096

    # Skill compiler (Transformer hypernetwork stub dims)
    compiler_layers: int = 4
    compiler_heads: int = 8
    max_skill_tokens: int = 4096

    pretrain_docs: int = 171_000
    pretrain_tokens_m: float = 300.0
    pretrain_epochs: int = 10
    pretrain_batch: int = 64
    pretrain_lr: float = 5e-5
    pretrain_warmup: int = 200
    pretrain_weight_decay: float = 0.1

    sft_alfworld_traj: int = 237
    sft_search_traj: int = 500
    sft_epochs: int = 10
    sft_batch: int = 32
    sft_lr: float = 1e-5
    sft_warmup: int = 400

    chunk_latent_frames: int = 9  # unused; kept for cross-stub parity
    alfworld_skills: int = 5
    search_qa_skills: int = 3

    # Token efficiency vs in-context (paper §4.2)
    prefill_reduction_alfworld: float = 0.641
    prefill_reduction_search_qa: float = 0.722

    control_layers: tuple[int, ...] = field(default_factory=tuple)  # not used; LatentSkill uses LoRA not ControlNet

    @property
    def injection_positions(self) -> int:
        return len(self.target_modules) * self.num_layers
