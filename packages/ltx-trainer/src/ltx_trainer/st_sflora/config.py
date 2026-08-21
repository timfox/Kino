"""ST-SFLora configuration (arXiv:2605.26120)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StSfloraConfig:
    paper_arxiv: str = "2605.26120"
    paper_title: str = (
        "Semantic-aware Token Selection and Resource Optimization for "
        "Communication-efficient Split Federated Fine-tuning in Edge Intelligence"
    )
    authors: str = "Xianke Qiang, Zheng Chang, Geyong Min"

    # ViT-B/16 default geometry (Sec. III)
    num_patch_tokens: int = 196
    embed_dim: int = 768
    batch_size: int = 64
    bits_per_element: int = 32
    client_cut_layers: int = 6

    # Wireless edge setting (Sec. VII-A)
    num_clients: int = 100
    bandwidth_total_hz: float = 50e6
    p_max_w: float = 0.2
    noise_psd_dbm_hz: float = -174.0
    path_loss_exponent: float = 2.5

    # Alternating optimization
    max_outer_iters: int = 8
    tol_power: float = 1e-4
    tol_bandwidth: float = 1e-4
    tol_k: float = 1.0
    tol_tau: float = 1e-6

    # Token / client bounds
    k_min: int = 64
    k_default_full: int = 196

    benchmarks: tuple[str, ...] = field(
        default_factory=lambda: (
            "ImageNet100",
            "Oxford Flowers-102",
            "CUB-200-2011",
        )
    )
    backbones: tuple[str, ...] = field(
        default_factory=lambda: ("ViT-S/16", "ViT-B/16", "ViT-L/16")
    )
    baselines: tuple[str, ...] = field(
        default_factory=lambda: (
            "LocalLoRA",
            "FedLoRA",
            "SplitLoRA",
            "SFLora",
            "ST-SFLora-Full",
            "ST-SFLora",
        )
    )
