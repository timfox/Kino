"""Configuration for PESD-ViT NAFLD multi-task histology (arXiv:2605.29852)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PesdVitConfig:
    paper_arxiv: str = "arXiv:2605.29852"
    backbone: str = "Swin-Tiny-22k"
    feature_dim: int = 768
    adapter_rank: int = 16
    num_tasks: int = 3
    task_names: tuple[str, ...] = ("steatosis", "ballooning", "inflammation")
    num_classes_per_task: tuple[int, ...] = (4, 3, 4)  # NAS ordinal bins
    ortho_lambda: float = 0.1
    task_manual_weights: tuple[float, ...] = (1.0, 1.2, 1.1)
    log_sigma_min: float = -4.0
    log_sigma_max: float = 2.0
    patch_input_size: int = 224
    patch_source_size: int = 416
    dataset_patches: int = 3192
    train_epochs: int = 200
    batch_size: int = 16
    lr: float = 8e-5
    weight_decay: float = 0.01
    fold_role: str = "nafld_nas_multitask_proxy"
