"""FashionLens + U-FIRE configuration (Wen et al., arXiv:2605.22552)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FashionLensConfig:
    """Hyperparameters from Sec. V-A (Qwen3-VL-4B backbone)."""

    paper_arxiv: str = "arXiv:2605.22552"
    code_url: str = "https://github.com/haokunwen/FashionLens"
    backbone: str = "Qwen3-VL-4B"
    embed_dim_D: int = 2560
    low_rank_d: int = 32
    ema_alpha: float = 0.9
    ggas_gamma: float = 0.5
    ggas_eta: float = 1.0
    min_sample_eps: float = 0.02
    beta_ortho: float = 1e-2
    beta_reg: float = 1e-4
    infonce_tau: float = 0.07
    lora_rank: int = 8
    train_epochs: int = 5
    lr: float = 1e-4
    gradcache_effective_batch: int = 64
    ufire_train_samples: int = 325_094
    ufire_val_samples: int = 36_187
    ufire_test_samples: int = 46_759
    ufire_num_datasets: int = 15
    ufire_train_tasks: int = 9
    ufire_ood_tasks: int = 2
