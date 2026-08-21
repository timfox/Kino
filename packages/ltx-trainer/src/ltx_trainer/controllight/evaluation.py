"""Framework card, evaluation demo, and multi-strength trajectory eval."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.controllight.benchmarks import benchmarks_bundle
from ltx_trainer.controllight.config import ControlLightConfig
from ltx_trainer.controllight.li_lpips import li_lpips_proxy, lwfm_ablation_smoke
from ltx_trainer.controllight.light100k import compare_interpolation_strategies, pair_passes_edge_filter
from ltx_trainer.controllight.metrics import clip_direction_score, delta_smooth
from ltx_trainer.controllight.retinex import alpha_blend_interpolate, build_light100k_group, retinex_interpolate
from ltx_trainer.controllight.train_step import training_step_demo


def framework_card(cfg: ControlLightConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ControlLightConfig()
    bench = benchmarks_bundle()
    return {
        "name": "ControlLight",
        "paper": cfg.paper_arxiv,
        "title": "Towards Controllable, Consistent, and Generalizable Low-Light Enhancement",
        "website": cfg.website,
        "authors": "Yufeng Yang, Jianzhuang Liu, Jisheng Chu, Yuqi Peng, Xianfang Zeng, Jiancheng Huang, Shifeng Chen",
        "institutions": ["SIAT CAS", "Zhejiang University"],
        "base_model": cfg.base_model,
        "dataset": {
            "name": "Light100K",
            "n_low_light_filtered": cfg.light100k_n_images,
            "n_pairs": cfg.light100k_n_pairs,
            "group": "G = {I0, I0.2, I0.4, I0.6, I0.8, I1}",
            "interpolation": "Retinex log-illumination + conservative reflectance (Eq. 1–2)",
        },
        "loss": {
            "standard": "L_FM (Eq. 3)",
            "proposed": "L_wFM misalignment-aware weighted FM (Eq. 6)",
            "weight_defaults": {
                "d_px": cfg.dist_threshold_px,
                "alpha": cfg.mask_alpha,
                "w_min": cfg.weight_min,
            },
        },
        "control": "LoRA strength s scales ΔW and selects pseudo-GT I_s (Eq. 7)",
        "training": {
            "resolution": cfg.train_resolution,
            "lora_rank": cfg.lora_rank,
            "lr": cfg.learning_rate,
            "steps": cfg.training_steps,
            "batch_size": cfg.global_batch_size,
        },
        "eval_strengths": list(cfg.eval_strengths),
        "restoration_prompt": cfg.restoration_prompt,
        "benchmarks": bench,
        "ltx_hook": (
            "Continuous strength s ∈ [0,1] for HDR/SDR bracket delivery; "
            "Light100K groups for LTX low-light clip conditioning"
        ),
    }


def knowledge_card(cfg: ControlLightConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ControlLightConfig()
    return {
        "framework": framework_card(cfg),
        "limitations": [
            "FLUX.2-klein-9B + 300M LoRA training is external to this repo",
            "CLIP-Dir and δ_smooth use proxies unless LPIPS/CLIP backends are wired",
            "Bilateral illumination smooth is Gaussian-separated proxy",
        ],
    }


def enhance_trajectory(
    i0: Tensor,
    i1: Tensor,
    *,
    strengths: tuple[float, ...] | None = None,
    use_retinex: bool = True,
) -> list[Tensor]:
    """Inference-time continuous enhancement via pseudo-pair interpolation (no FLUX)."""
    cfg = ControlLightConfig()
    sts = strengths or cfg.eval_strengths
    group = build_light100k_group(i0, i1, use_retinex=use_retinex)
    out: list[Tensor] = []
    for s in sts:
        key = min(group.keys(), key=lambda k: abs(k - float(s)))
        out.append(group[key])
    return out


def evaluation_demo(cfg: ControlLightConfig | None = None, *, seed: int = 7, size: int = 48) -> dict[str, Any]:
    cfg = cfg or ControlLightConfig()
    torch.manual_seed(seed)
    i0 = torch.rand(3, size, size) * 0.1
    i1 = torch.rand(3, size, size) * 0.5 + 0.4

    mid_r = retinex_interpolate(i0, i1, 0.5)
    mid_a = alpha_blend_interpolate(i0, i1, 0.5)
    interp_cmp = compare_interpolation_strategies(i0, i1, cfg=cfg)
    traj = enhance_trajectory(i0, i1, strengths=cfg.eval_strengths)
    lwfm = lwfm_ablation_smoke(i0, i1)
    train = training_step_demo(seed=seed, size=size)
    li = li_lpips_proxy(i0, mid_r)

    return {
        "paper": cfg.paper_arxiv,
        "edge_filter_pass": pair_passes_edge_filter(i0, i1),
        "retinex_brighter_than_alpha_at_s05": float(mid_r.mean() > mid_a.mean()),
        "interpolation_ablation": interp_cmp,
        "trajectory": {
            "strengths": list(cfg.eval_strengths),
            "delta_smooth": delta_smooth(traj),
            "clip_dir_proxy": clip_direction_score(traj),
            "brightness_means": [float(t.mean().item()) for t in traj],
        },
        "lwfm_ablation": lwfm,
        "training_step": train,
        "li_lpips_retinex_mid": li,
        "table4_anchors": benchmarks_bundle()["table4_lwfm"],
    }


def evaluation_smoke(cfg: ControlLightConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ControlLightConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.controllight",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": (
            ev["retinex_brighter_than_alpha_at_s05"]
            and ev["trajectory"]["clip_dir_proxy"] > 0
            and ev["training_step"]["loss_LwFM"] == ev["training_step"]["loss_LwFM"]
        ),
        "keys": list(ev.keys())[:8],
        "trajectory_clip_dir": ev["trajectory"]["clip_dir_proxy"],
    }
