"""Fine-grained vehicle classification pipeline (Padmanaban & Feng, arXiv:2606.05149)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FgVehicleVitConfig:
    paper_arxiv: str = "arXiv:2606.05149"
    title: str = (
        "An Open-Source Two-Stage Computer Vision Pipeline for "
        "Fine-Grained Vehicle Classification using Vision Transformers"
    )
    repo_url: str = "https://github.com/[repo]"  # placeholder per paper §6
    open_data_url: str = "https://fenggroup.org/facilities/lidar-bike.html#open-data-repository"

    # Stage 1 — RT-DETR
    rtdetr_checkpoint: str = "PekingU/rtdetr_r50vd_coco_o365"
    stage1_conf_threshold: float = 0.35
    coco_vehicle_class_ids: tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck
    min_box_short_side_frac: float = 0.055
    min_box_area_frac: float = 0.0004

    # Stage 2 — ViT-Base/16
    vit_checkpoint: str = "google/vit-base-patch16-224-in21k"
    num_classes: int = 6
    train_image_size: int = 256
    crop_size: int = 224
    abstention_threshold: float = 0.60

    # Training (§2.5)
    focal_gamma: float = 2.0
    epochs: int = 30
    lr: float = 2e-5
    batch_size: int = 64
    training_samples: int = 16_581

    # Evaluation sets (§2.7, §3)
    eval_in_distribution_n: int = 3_805
    eval_out_of_distribution_n: int = 311
    eval_id_accuracy: float = 0.94
    eval_ood_accuracy: float = 0.89

    fine_grained_classes: tuple[str, ...] = field(
        default_factory=lambda: (
            "passenger car",
            "SUV",
            "pickup truck",
            "minivan",
            "large van",
            "commercial truck",
        )
    )
