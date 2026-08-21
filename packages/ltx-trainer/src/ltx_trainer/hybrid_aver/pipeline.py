"""Framework card, AVE tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.hybrid_aver.config import HybridAverConfig
from ltx_trainer.hybrid_aver.fusion import HybridAverFusionStub


def framework_card(cfg: HybridAverConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HybridAverConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "dataset": cfg.dataset,
        "backbones": {
            "video": cfg.video_backbone,
            "audio": cfg.audio_backbone,
            "frozen": True,
            "cached_features": True,
        },
        "fusion_stack": [
            "FiLM audio conditioning",
            "bidirectional cross-attention",
            "multimodal Transformer encoder",
            "modality-temporal attention (MAF)",
        ],
        "segments_T": cfg.num_segments,
        "hidden_dim": cfg.embed_dim,
        "seeds": 5,
        "headline": {
            "best_val_acc_peak": cfg.best_val_acc_peak,
            "test_acc": f"{cfg.test_acc_mean:.4f} ± {cfg.test_acc_std:.4f}",
        },
    }


def table1_multi_seed() -> list[dict[str, Any]]:
    """Table I — five-seed AVE comparison."""
    return [
        {
            "model": "Proposed Hybrid Fusion",
            "best_val_acc": "0.8948 ± 0.0084",
            "test_acc": "0.8385 ± 0.0140",
            "balanced_acc": "0.8277 ± 0.0168",
            "weighted_f1": "0.8368 ± 0.0145",
        },
        {
            "model": "Simple AV Fusion",
            "best_val_acc": "0.8843 ± 0.0146",
            "test_acc": "0.8239 ± 0.0140",
            "balanced_acc": "0.8228 ± 0.0231",
            "weighted_f1": "0.8215 ± 0.0152",
        },
        {
            "model": "Audio-only AST",
            "best_val_acc": "0.8670 ± 0.0085",
            "test_acc": "0.7991 ± 0.0100",
            "balanced_acc": "0.7725 ± 0.0127",
            "weighted_f1": "0.7935 ± 0.0092",
        },
        {
            "model": "Video-only VideoMAE",
            "best_val_acc": "0.6035 ± 0.0135",
            "test_acc": "0.5060 ± 0.0215",
            "balanced_acc": "0.5038 ± 0.0304",
            "weighted_f1": "0.5024 ± 0.0229",
        },
    ]


def table2_compute() -> list[dict[str, Any]]:
    """Table II — params and training time."""
    return [
        {"model": "Audio-only AST", "params_M": 0.804381, "train_h": 0.5175},
        {"model": "Video-only VideoMAE", "params_M": 0.804381, "train_h": 0.2923},
        {"model": "Simple AV Fusion", "params_M": 2.643, "train_h": 0.3134},
        {"model": "Proposed Hybrid Fusion", "params_M": 6.856, "train_h": 0.3835},
    ]


def class_balanced_ce(logits: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
    """Eq. (14) stub."""
    return F.cross_entropy(logits, labels, weight=weights)


def forward_smoke(cfg: HybridAverConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HybridAverConfig()
    model = HybridAverFusionStub(cfg)
    b, t, d = cfg.demo_batch, cfg.demo_time_steps, cfg.embed_dim
    v = torch.randn(b, t, d)
    a = torch.randn(b, t, d)
    out = model(v, a)
    labels = torch.randint(0, cfg.num_classes, (b,))
    w = torch.ones(cfg.num_classes)
    loss = class_balanced_ce(out["logits"], labels, w)
    return {
        "logits_shape": list(out["logits"].shape),
        "fused_shape": list(out["fused"].shape),
        "loss": float(loss.detach()),
        "trainable_params_m": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    }


def evaluation_demo(cfg: HybridAverConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HybridAverConfig()
    return {
        "framework": framework_card(cfg),
        "table1": table1_multi_seed(),
        "table2": table2_compute(),
        "splits": {"train": cfg.split_train, "val": cfg.split_val, "test": cfg.split_test},
        "forward": forward_smoke(cfg),
    }
