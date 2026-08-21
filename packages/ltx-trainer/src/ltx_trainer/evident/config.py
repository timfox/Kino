"""EVIDENT cross-domain VTG (Ahn et al., arXiv:2605.26104)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EVIDENTConfig:
    """Defaults from paper Sec. 4–5 and Table 4 (appendix)."""

    backbone: str = "Qwen2.5-VL-7B"
    hidden_dim: int = 3584
    bottleneck_dim: int = 896
    num_slots: int = 4
    slot_iterations: int = 3
    tokens_per_frame: int = 64
    adapter_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(11)))
    lora_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(11, 28)))
    lora_rank: int = 16
    lora_alpha: int = 64
    eb_distill_weight: float = 0.1
    num_frames_charades: int = 20
    num_frames_qvh: int = 60
    resolution: int = 224
    trainable_params_m: float = 201.0
    learning_rate: float = 5e-5

    datasets: tuple[str, ...] = ("charades_sta", "qvhighlights", "didemo")
