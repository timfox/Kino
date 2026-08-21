"""TORM framework card and reference experiment tables."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ltx_trainer.torm.config import TORMConfig
from ltx_trainer.torm.latent import LatentRolloutStub, pool_thought_video_targets
from ltx_trainer.torm.training import stage1_loss, stage2_loss


def framework_card(cfg: TORMConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TORMConfig()
    return {
        "name": "TORM",
        "paper": "arXiv:2605.26014",
        "title": "Internalized Modeling for Spatial-Temporal Reasoning in Video-Language Models",
        "backbone": cfg.backbone,
        "latent_slots": cfg.num_latent_slots,
        "stages": [
            "Stage I: thought-video latent alignment + answer CE (λ=0.1)",
            "Stage II: answer-only consolidation (Coconut-style)",
        ],
        "inference": "bounded latent rollout; no thought-video / tools at test time",
        "repo": "https://github.com/aiming-lab/storm",
    }


def table_general_benchmarks() -> dict[str, dict[str, float]]:
    """Table 1 — VideoMME, MVBench, TempCompass (32 frames)."""
    return {
        "torm": {"videomme": 61.0, "mvbench": 61.1, "tempcompass": 74.3},
        "qwen_sft": {"videomme": 55.4, "mvbench": 60.5, "tempcompass": 69.9},
        "qwen_cot": {"videomme": 60.8, "mvbench": 42.4, "tempcompass": 72.8},
        "video_thinker": {"videomme": 60.0, "mvbench": 62.4, "tempcompass": 67.8},
        "longvt": {"videomme": 59.7, "mvbench": 53.9, "tempcompass": 68.3},
    }


def table_reasoning_benchmarks() -> dict[str, dict[str, float]]:
    """Table 2 — reasoning-heavy benchmarks."""
    return {
        "torm": {"videoespresso": 58.7, "video_holmes": 37.8, "mmvu": 65.9},
        "qwen_sft": {"videoespresso": 49.6, "video_holmes": 31.4, "mmvu": 62.1},
        "video_thinker": {"videoespresso": 56.4, "video_holmes": 43.1, "mmvu": 63.4},
        "longvt": {"videoespresso": 52.7, "video_holmes": 35.5, "mmvu": 61.4},
    }


def table_3b_backbone() -> dict[str, dict[str, float]]:
    """Table 3 — Qwen2.5-VL-3B scalability."""
    return {
        "baseline": {"videomme": 50.9, "tempcompass": 55.0, "mmvu": 42.0},
        "torm": {"videomme": 55.4, "tempcompass": 67.5, "mmvu": 58.5},
    }


def table_latency_mmvu() -> dict[str, dict[str, float]]:
    """Table 4 — MMVU throughput / latency."""
    return {
        "longvt": {"throughput_its": 0.065, "latency_s": 15.44},
        "video_thinker": {"throughput_its": 0.058, "latency_s": 17.20},
        "torm": {"throughput_its": 2.14, "latency_s": 0.47},
    }


def table_latent_size_ablation() -> dict[int, dict[str, float]]:
    """Table 5 — latent token budget."""
    return {
        4: {"videomme": 58.9, "tempcompass": 71.1, "mvbench": 55.0},
        8: {"videomme": 61.0, "tempcompass": 74.3, "mvbench": 61.1},
        16: {"videomme": 52.2, "tempcompass": 67.3, "mvbench": 55.5},
    }


def table_two_stage_ablation() -> dict[str, dict[str, float]]:
    """Table 6 — training stage ablation."""
    return {
        "stage_i_only": {"videomme": 58.8, "tempcompass": 72.0, "mvbench": 56.0},
        "stage_ii_only": {"videomme": 56.6, "tempcompass": 72.6, "mvbench": 56.8},
        "full_torm": {"videomme": 61.0, "tempcompass": 74.3, "mvbench": 61.1},
    }


def table_same_video_retrieval() -> dict[str, dict[str, float]]:
    """Figure 6 — same-video retrieval probe."""
    return {
        "latent": {"hit_at_1": 71.0, "hit_at_5": 79.0, "mrr": 75.0},
        "text": {"hit_at_1": 29.0, "hit_at_5": 71.0, "mrr": 50.0},
        "random": {"hit_at_1": 0.0, "hit_at_5": 0.0, "mrr": 0.0},
    }


def training_step_demo(
    cfg: TORMConfig | None = None,
    *,
    stage: int = 1,
    batch_size: int = 2,
    device: str = "cpu",
) -> dict[str, float]:
    """Smoke: latent rollout + Stage I/II loss."""
    cfg = cfg or TORMConfig()
    dev = torch.device(device)
    b = batch_size
    k, d = cfg.num_latent_slots, cfg.hidden_dim

    rollout = LatentRolloutStub(cfg).to(dev)
    context = torch.randn(b, d, device=dev)
    z = rollout(context)

    thought = torch.randn(b, 48, d, device=dev)
    g = torch.stack([pool_thought_video_targets(thought[i], k) for i in range(b)])

    head = nn.Linear(d, 32).to(dev)
    ans_logits = head(torch.randn(b, 6, d, device=dev))
    ans_labels = torch.randint(0, 32, (b, 6), device=dev)

    if stage == 1:
        _, metrics = stage1_loss(z, g, ans_logits, ans_labels, cfg=cfg)
        return {**metrics, "num_latent_slots": float(k), "stage": 1.0}
    loss = stage2_loss(ans_logits, ans_labels)
    return {"l_total": float(loss.item()), "num_latent_slots": float(k), "stage": 2.0}


def evaluation_demo(*, device: str = "cpu") -> dict[str, Any]:
    s1 = training_step_demo(stage=1, device=device)
    gen = table_general_benchmarks()
    lat = table_latency_mmvu()
    return {
        **s1,
        "torm_videomme": gen["torm"]["videomme"],
        "sft_videomme": gen["qwen_sft"]["videomme"],
        "videomme_gain": gen["torm"]["videomme"] - gen["qwen_sft"]["videomme"],
        "torm_mmvu": table_reasoning_benchmarks()["torm"]["mmvu"],
        "latency_s_item": lat["torm"]["latency_s"],
        "longvt_latency_s": lat["longvt"]["latency_s"],
        "latent_hit_at_1": table_same_video_retrieval()["latent"]["hit_at_1"],
    }
