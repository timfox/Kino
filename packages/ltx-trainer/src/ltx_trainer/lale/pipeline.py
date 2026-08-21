"""LALE framework card, tables, and evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.lale.config import LaleConfig
from ltx_trainer.lale.metrics import dice_loss, f1_from_logits
from ltx_trainer.lale.model import LALE, count_parameters_m


def framework_card(cfg: LaleConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LaleConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "dataset": cfg.dataset,
        "architecture": {
            "stem": "2× 3×3 stride-2 conv + RMSNorm + StarReLU → H/4",
            "encoder": "Stages 1–2 ConvMixer (local); 3–4 Transformer (global)",
            "decoder": f"All-MLP multi-scale, C_dec={cfg.decoder_channels}",
        },
        "efficiency_ops": ["RMSNorm", "StarReLU", "3×3 stride-2 downsampling"],
        "headline": {
            "lale_s1_params_m": cfg.lale_s1_params_m,
            "lale_s1_f1": cfg.lale_s1_f1,
            "upernet_f1": cfg.upernet_f1,
            "f1_gap": cfg.f1_gap_to_upernet,
            "param_ratio_vs_upernet": round(cfg.upernet_params_m / cfg.lale_s1_params_m, 1),
            "gmac_ratio_vs_upernet": round(cfg.upernet_gmacs / cfg.lale_s1_gmacs, 1),
        },
        "training": {"loss": cfg.loss, "images": cfg.train_images},
    }


def table1_aras400k_benchmark() -> list[dict[str, Any]]:
    """Paper Table 1 — selected CNN, LALE, and transformer rows."""
    return [
        {"architecture": "UPerNet", "f1": 77.31, "params_m": 11.6, "gmacs": 13.62, "throughput": 89889, "size_mb": 44.49},
        {"architecture": "Unet", "f1": 77.23, "params_m": 6.3, "gmacs": 3.05, "throughput": 80002, "size_mb": 24.02},
        {"architecture": "Segformer", "f1": 76.47, "params_m": 4.5, "gmacs": 2.05, "throughput": 85491, "size_mb": 17.23},
        {"architecture": "LALE-S1", "f1": 74.69, "params_m": 1.6, "gmacs": 0.59, "throughput": 160253, "size_mb": 5.98},
        {"architecture": "LALE-S2", "f1": 75.88, "params_m": 2.6, "gmacs": 0.78, "throughput": 117105, "size_mb": 9.97},
        {"architecture": "EffFormer-L7", "f1": 75.35, "params_m": 100.3, "gmacs": 32.16, "throughput": 2691, "size_mb": 383.22},
        {"architecture": "DeiT3-Base", "f1": 76.10, "params_m": 117.1, "gmacs": 39.89, "throughput": 6156, "size_mb": 446.64},
    ]


def table2_ablation() -> list[dict[str, Any]]:
    """Paper Table 2 — kernel / pretrain ablation (selected)."""
    return [
        {"config": "S2-K3-PT", "f1": 75.88, "params_m": 2.6, "gmacs": 0.78, "throughput": 117105},
        {"config": "S1-K3-PT", "f1": 74.69, "params_m": 1.6, "gmacs": 0.59, "throughput": 160253},
        {"config": "S2-K3", "f1": 73.40, "params_m": 2.6, "gmacs": 0.78, "throughput": 118265},
        {"config": "B-S2-K7", "f1": 71.67, "params_m": 2.7, "gmacs": 0.94, "throughput": 159305},
    ]


def table3_lits() -> list[dict[str, Any]]:
    """Paper Table 3 — LiTS liver / tumor F1 (selected)."""
    return [
        {"architecture": "UPerNet", "liver_f1": 94.69, "tumor_f1": 71.47},
        {"architecture": "Unet", "liver_f1": 95.06, "tumor_f1": 79.67},
        {"architecture": "LALE-S1", "liver_f1": 92.62, "tumor_f1": 71.37},
        {"architecture": "LALE-S2", "liver_f1": 93.62, "tumor_f1": 69.67},
        {"architecture": "LALE-S4", "liver_f1": 93.98, "tumor_f1": 74.67},
    ]


def forward_smoke(scale: str = "S1") -> dict[str, Any]:
    cfg = LaleConfig()
    model = LALE(cfg, scale=scale)
    model.eval()
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    y = torch.randint(0, cfg.num_classes, (2, cfg.image_size, cfg.image_size))
    with torch.no_grad():
        logits = model(x)
    loss = dice_loss(logits, y, cfg.num_classes)
    f1 = f1_from_logits(logits, y, cfg.num_classes)
    return {
        "scale": scale,
        "logits_shape": list(logits.shape),
        "params_m": round(count_parameters_m(model), 3),
        "dice_loss": float(loss),
        "f1_proxy": f1,
    }


def evaluation_demo(cfg: LaleConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LaleConfig()
    return {
        "framework": framework_card(cfg),
        "table1_aras400k": table1_aras400k_benchmark(),
        "table2_ablation": table2_ablation(),
        "table3_lits": table3_lits(),
        "forward_s1": forward_smoke("S1"),
        "forward_s2": forward_smoke("S2"),
        "pareto_note": "LALE-S1/S2 on accuracy-efficiency frontier vs CNN and ViT encoders",
    }
