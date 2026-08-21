"""SSL art classification on WikiArt (Melis et al., IRCDL'26)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SSLArtConfig:
    venue: str = "IRCDL'26"
    dataset: str = "WikiArt"
    dataset_size: int = 80_000
    style_classes: int = 27
    genre_classes: int = 10  # 11 minus Unknown Genre
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10
    supervised_baseline: str = "EfficientNetV2-L"
    clip_backbone: str = "CLIP-ViT-L/14"
    dino_backbone: str = "DINOV3-ViT-L/16"
    knn_k: int = 1
    linear_lr: float = 1e-4
    linear_weight_decay: float = 1e-4
    linear_batch_size: int = 1024
    linear_max_epochs: int = 100
    linear_early_stop_patience: int = 5
    random_seed: int = 42
