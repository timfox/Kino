"""Paris 2.0 framework card and reference experiment tables."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.paris2.config import Paris2Config
from ltx_trainer.paris2.experts import ExpertPool, top_k_mask
from ltx_trainer.paris2.flow_matching import flow_matching_loss, sample_linear_path
from ltx_trainer.paris2.router import Paris2Router


def framework_card(cfg: Paris2Config | None = None) -> dict[str, Any]:
    cfg = cfg or Paris2Config()
    return {
        "name": "Paris 2.0",
        "paper": "arXiv:2605.26064",
        "title": "A Decentralized Diffusion Model for Video Generation",
        "predecessor": "Paris 1.0 (arXiv:2510.03434)",
        "organization": "Bagel Labs",
        "architecture": {
            "vae": cfg.vae,
            "experts": f"{cfg.num_experts}×{cfg.expert_params_b}B {cfg.expert_backbone}",
            "router": f"DiT-B ~{cfg.router_params_m}M params",
            "text": list(cfg.text_encoders),
        },
        "training": "Experts trained per cluster without sync; router as cluster classifier",
        "inference": f"top-{cfg.top_k_experts} routing per denoising step + ODE solver",
    }


def table_stage1_t2v() -> dict[str, dict[str, float | str]]:
    """Table 1 — iso-FLOP low-res text-to-video (N=2048)."""
    return {
        "ddm": {
            "fvd": 279.01,
            "clip_text_video": 0.2178,
            "clip_std": 0.0012,
            "aesthetic": 3.9036,
            "aesthetic_std": 0.0082,
            "motion_px_per_frame": 0.712,
            "motion_std": 0.057,
        },
        "monolithic": {
            "fvd": 561.04,
            "clip_text_video": 0.2032,
            "clip_std": 0.0011,
            "aesthetic": 3.7950,
            "aesthetic_std": 0.0077,
            "motion_px_per_frame": 0.555,
            "motion_std": 0.043,
        },
    }


def table_relative_improvement() -> dict[str, float]:
    """Figure 2 — relative gain of DDM over monolithic (higher bar = larger gain)."""
    mono = table_stage1_t2v()["monolithic"]
    ddm = table_stage1_t2v()["ddm"]
    return {
        "fvd_improvement_ratio": mono["fvd"] / ddm["fvd"],  # ~2.01× lower is better
        "clip_delta": ddm["clip_text_video"] - mono["clip_text_video"],
        "aesthetic_delta": ddm["aesthetic"] - mono["aesthetic"],
    }


def table_expert_specialization() -> dict[str, float]:
    """Sec. 4.3 — in- vs out-of-cluster CLIP on assigned prompts."""
    return {
        "in_cluster_clip": 0.2175,
        "out_cluster_clip": 0.1781,
        "clip_gap": 0.039,
    }


def table_switching_schedule_ablation() -> dict[str, Any]:
    """Figure 5 — manual two-expert schedules (N=40 prompts, router bypassed)."""
    return {
        "prompts_prefer_switching": 24,
        "prompts_total": 40,
        "alternating_beats_single_expert": True,
        "note": "Expert A vs B order asymmetry indicates high-noise vs low-noise specialization",
    }


def training_step_demo(
    cfg: Paris2Config | None = None,
    *,
    batch_size: int = 2,
    device: str = "cpu",
) -> dict[str, float]:
    """Smoke: expert flow-matching + router cluster classification."""
    cfg = cfg or Paris2Config()
    dev = torch.device(device)
    b = batch_size
    c, f, h, w = cfg.latent_channels, cfg.latent_frames, cfg.latent_height, cfg.latent_width

    x0 = torch.randn(b, c, f, h, w, device=dev)
    t = torch.rand(b, device=dev)
    x_t, v_target = sample_linear_path(x0, t)

    pool = ExpertPool(cfg).to(dev)
    router = Paris2Router(cfg).to(dev)
    clip = torch.randn(b, cfg.clip_dim, device=dev)

    weights = router(x_t, t, clip)
    weights = top_k_mask(weights, cfg.top_k_experts)
    v_pred = pool(x_t, t, weights)
    loss_flow = float(flow_matching_loss(v_pred, v_target).item())

    labels = torch.randint(0, cfg.num_experts, (b,), device=dev)
    logits = router.cluster_logits(x_t, t, clip)
    loss_router = float(F.cross_entropy(logits, labels).item())

    return {
        "flow_loss": loss_flow,
        "router_loss": loss_router,
        "mean_routing_entropy": float(
            -(weights * weights.clamp(min=1e-8).log()).sum(dim=-1).mean()
        ),
        "num_experts": float(cfg.num_experts),
    }


def evaluation_demo(*, device: str = "cpu") -> dict[str, Any]:
    step = training_step_demo(device=device)
    tab = table_stage1_t2v()
    rel = table_relative_improvement()
    spec = table_expert_specialization()
    return {
        **step,
        "ddm_fvd": tab["ddm"]["fvd"],
        "monolithic_fvd": tab["monolithic"]["fvd"],
        "fvd_ratio": rel["fvd_improvement_ratio"],
        "ddm_clip": tab["ddm"]["clip_text_video"],
        "monolithic_clip": tab["monolithic"]["clip_text_video"],
        "expert_in_cluster_clip": spec["in_cluster_clip"],
        "expert_out_cluster_clip": spec["out_cluster_clip"],
    }
