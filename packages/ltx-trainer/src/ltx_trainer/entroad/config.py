"""Configuration for EntroAD (arXiv:2605.28630)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EntroADConfig:
    paper_arxiv: str = "arXiv:2605.28630"
    backbone: str = "CLIP ViT-L/14@336px"
    image_size: int = 518
    train_source: str = "MVTec-AD"
    test_datasets: tuple[str, ...] = (
        "MVTec-AD",
        "VisA",
        "BTAD",
        "DTD",
        "MPDD",
        "BrainMRI",
        "HeadCT",
        "Br35H",
        "Endo",
        "Kvasir",
    )
    attention_layers: tuple[int, ...] = (6, 12, 18, 24)
    router_temperature: float = 0.1
    gate_tau: float = 0.5
    gate_k0: float = 5.0
    gate_k1: float = 50.0
    lambda_a: float = 0.7
    lambda_b: float = 0.3
    fusion_alpha: float = 0.7
    fusion_beta: float = 0.3
    inference_k: float = 0.7
    lambda_dice: float = 1.0
    stage1_epochs: int = 1
    stage2_epochs: int = 5
    fold_role: str = "zero_shot_anomaly_proxy"
